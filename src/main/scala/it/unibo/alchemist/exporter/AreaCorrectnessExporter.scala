package it.unibo.alchemist.exporter

import it.unibo.alchemist.boundary.extractors.AbstractDoubleExporter
import it.unibo.Utils._
import it.unibo.alchemist.model._
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.scafi.Sensors

import java.{lang, util}

class AreaCorrectnessExporter extends AbstractDoubleExporter {

  private lazy val leaderMolecule = Sensors.leaderId
  private lazy val areaIdMolecule = Sensors.areaId
//  private lazy val labelsMolecule = Sensors.labels

  override def getColumnNames: util.List[String] =
    util.List.of("AreaCorrectness")

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: Time,
      l: Long
  ): util.Map[String, lang.Double] = {
    val nodes = environment.getNodesAsScala
    val areasCorrectness = nodes
      .map(node => new SimpleNodeManager[T](node))
      .map(node => node.get[Int](leaderMolecule) -> node.get[Int](areaIdMolecule))
      .groupBy(_._1)
      .map { case (leaderId, values) => leaderId -> values.map(_._2) }
      .map { case (leaderId, areaIds) => leaderId -> (environment.getNodeByID(leaderId).getConcentration(areaIdMolecule).asInstanceOf[Int], areaIds) }
      .map { case (_, (areaId, areaIds)) => areaIds.foldLeft(0)((acc, elem) => if (elem == areaId) acc else acc + 1) }
      .sum

    util.Map.of(
      "AreaCorrectness",
      areasCorrectness.toDouble
    )
  }

  private def getLeaderLabels[T](node: Node[T], labels: Set[Int], labelsMolecule: String): (Int, Double) = {
    val leaderNode = node.manager
    val leaderLabels = leaderNode.getOrElse(labelsMolecule, Set.empty[Int])
    val labelUnion = labels.union(leaderLabels)
    val correctness = leaderLabels.size.toDouble / labelUnion.size
    node.getId -> correctness
  }
}
