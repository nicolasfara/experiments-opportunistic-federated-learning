package it.unibo.alchemist.exporter

import it.unibo.alchemist.boundary.extractors.AbstractDoubleExporter
import it.unibo.Utils._
import it.unibo.alchemist.model._
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.scafi.Sensors

import java.{lang, util}

class AreaCorrectness extends AbstractDoubleExporter {

  private lazy val leaderMolecule = Sensors.leaderId
  private lazy val labelsMolecule = Sensors.labels

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
      .map(node => (node.get[Int](leaderMolecule), node.getOrElse(labelsMolecule, Set.empty[Int])))
      .groupBy(_._1)
      .map { case (leaderId, labels) => leaderId -> labels.flatMap(_._2).toSet }
      .map { case (leaderId, labels) => getLeaderLabels(environment.getNodeByID(leaderId), labels, labelsMolecule) }

    util.Map.of(
      "AreaCorrectness",
      areasCorrectness.values.sum / areasCorrectness.size
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
