package it.unibo.alchemist.exporter

import it.unibo.Utils.EnvironmentOps
import it.unibo.alchemist.boundary.extractors.AbstractDoubleExporter
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.{Actionable, Environment, Time}
import it.unibo.scafi.Sensors

import java.{util, lang}

class AverageLossExporter extends AbstractDoubleExporter {
  override def getColumnNames: util.List[String] =
    util.List.of("AverageLossTraining", "AverageLossValidation")

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: Time,
      l: Long
  ): util.Map[String, lang.Double] = {
    val nodes = environment.getNodesAsScala
    val averagePerArea = nodes
      .map(node => new SimpleNodeManager[T](node))
      .map(node =>
        (
          node.get[Int](Sensors.areaId),
          node.getOrElse(Sensors.trainLoss, Double.NaN),
          node.getOrElse(Sensors.validationLoss, Double.NaN)
        )
      )
      .groupBy { case (areaId, _, _) => areaId }
      .map { case (id, losses) => id -> losses.map(e => (e._2, e._3)) }
      .map { case (id, losses) => id -> meanLosses(losses) }

    val (avgLossTraining, avgLossValidation) = meanLosses(
      averagePerArea.values.toList
    )

    util.Map.of(
      "AverageLossTraining",
      avgLossTraining,
      "AverageLossValidation",
      avgLossValidation
    )
  }

  private def meanLosses(losses: List[(Double, Double)]): (Double, Double) = {
    val (avgLossTraining, avgLossValidation) =
      losses.foldLeft((0.0, 0.0))((acc, e) => (acc._1 + e._1, acc._2 + e._2))
    (avgLossTraining / losses.size, avgLossValidation / losses.size)
  }
}
