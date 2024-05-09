package it.unibo.alchemist.exporter

import it.unibo.Utils.{EnvironmentOps, RichNode}
import it.unibo.alchemist.boundary.extractors.AbstractDoubleExporter
import it.unibo.alchemist.model.{Actionable, Environment, Time}
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.scafi.Sensors

import java.{lang, util}

class MoveInformationExporter extends AbstractDoubleExporter {
  override def getColumnNames: util.List[String] =
    util.List.of("AreaId", "LeaderId", "ValidationLoss", "ValidationAccuracy", "SameLeader")

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: Time,
      l: Long
  ): util.Map[String, lang.Double] = {

    val nodes = environment.getNodesAsScala
    val Some(node) =
      nodes
        .map(_.manager)
        .find(_.has(Sensors.movable))
    // .map(node => (node.get[Integer](Sensors.areaId), node.get[Integer](Sensors.leaderId)))
    util.Map.of(
      "AreaId",
      node.get[Integer](Sensors.areaId).toDouble,
      "LeaderId",
      node.get[Integer](Sensors.leaderId).toDouble,
      "ValidationLoss",
      node.getOrElse(Sensors.validationLoss, Double.NaN),
      "ValidationAccuracy",
      node.getOrElse(Sensors.validationAccuracy, Double.NaN),
      "SameLeader",
      if (node.get[Boolean](Sensors.sameLeader)) 1.0 else 0.0
    )
  }
}
