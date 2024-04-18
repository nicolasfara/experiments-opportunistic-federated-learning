package it.unibo.alchemist.exporter

import it.unibo.alchemist.boundary.extractors.AbstractDoubleExporter
import it.unibo.alchemist.model._
import it.unibo.alchemist.model.molecules.SimpleMolecule

import java.{lang, util}
import scala.jdk.CollectionConverters.IteratorHasAsScala

class AreaCorrectness extends AbstractDoubleExporter {

  lazy val leaderMolecule = new SimpleMolecule("leader")
  lazy val labelsMolecule = new SimpleMolecule("labels")

  override def getColumnNames: util.List[String] =
    util.List.of("AreaCorrectness")

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: Time,
      l: Long
  ): util.Map[String, lang.Double] = {
    val nodes = environment.getNodes.iterator().asScala.toList
    val areasCorrectness = nodes
      .map(node =>
        (
          node.getConcentration(leaderMolecule).asInstanceOf[Int],
          if (node.contains(labelsMolecule))
            node
              .getConcentration(new SimpleMolecule("labels"))
              .asInstanceOf[Set[Int]]
          else Set.empty[Int]
        )
      )
      .groupBy(_._1)
      .map { case (leaderId, labels) => leaderId -> labels.flatMap(_._2).toSet }
      .map { case (leaderId, labels) =>
        val leaderNode = environment.getNodeByID(leaderId)
        val leaderLabels =
          if (leaderNode.contains(labelsMolecule))
            environment
              .getNodeByID(leaderId)
              .getConcentration(labelsMolecule)
              .asInstanceOf[Set[Int]]
          else Set.empty[Int]
        val labelUnion = labels.union(leaderLabels)
        val correctness = labels.size.toDouble / labelUnion.size
        leaderId -> correctness
      }
    util.Map.of(
      "AreaCorrectness",
      areasCorrectness.values.sum / areasCorrectness.size
    )
  }
}
