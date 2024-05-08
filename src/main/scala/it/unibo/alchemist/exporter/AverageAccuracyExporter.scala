package it.unibo.alchemist.exporter

import it.unibo.Utils.{EnvironmentOps, RichNode}
import it.unibo.alchemist.boundary.extractors.AbstractDoubleExporter
import it.unibo.alchemist.model.{Actionable, Environment, Time}
import it.unibo.scafi.Sensors
import it.unibo.scafi.Sensors.{leaderId, validationAccuracy}

import java.{lang, util}

class AverageAccuracyExporter extends AbstractDoubleExporter {

  override def getColumnNames: util.List[String] =
    util.List.of("AverageAccuracy")

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: Time,
      l: Long
  ): util.Map[String, lang.Double] = {
    val nodes = environment.getNodesAsScala
    val averageAccuracyPerArea = nodes
      .map(node => node.manager)
      .map(node => node.get[Int](Sensors.areaId) -> node.getOrElse[Double](validationAccuracy, Double.NaN))
      .groupBy(_._1)
      .map { case (_, accuracies) => accuracies.map(_._2).sum / accuracies.size }

    util.Map.of(
      "AverageAccuracy",
      averageAccuracyPerArea.sum / averageAccuracyPerArea.size
    )
  }
}
