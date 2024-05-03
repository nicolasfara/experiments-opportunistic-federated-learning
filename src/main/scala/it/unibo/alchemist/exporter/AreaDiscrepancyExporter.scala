package it.unibo.alchemist.exporter

import it.unibo.alchemist.boundary.extractors.AbstractDoubleExporter
import it.unibo.alchemist.model.{Actionable, Environment, Time}

import java.{lang, util}
import scala.jdk.CollectionConverters.{MapHasAsJava, SeqHasAsJava}
import it.unibo.Utils._
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.scafi.Sensors
import it.unibo.scafi.interop.PythonModules.utils
import me.shadaj.scalapy.py
class AreaDiscrepancyExporter extends AbstractDoubleExporter {

  override def getColumnNames: util.List[String] = util.List.of("Discrepancy") // todo

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: Time,
      l: Long
  ): util.Map[String, lang.Double] = {
    val nodes = environment.getNodesAsScala.map(_.manager)
    val nodesPerAreas =
      nodes.groupBy(node => node.get[Int](Sensors.areaId))
    val areasWithMeanDiscrepancy = nodesPerAreas
      .map { case (areas, nodes) =>
        areas -> (nodes.minBy(_.node.getId) -> nodes)
      }
      .map { case (areas, (firstNode, nodes)) =>
        areas -> nodes
          .map(node => discrepancy(firstNode, node))
          .sum / nodes.size
      }
    util.Map.of(
      "Discrepancy",
      if (areasWithMeanDiscrepancy.nonEmpty) {
        areasWithMeanDiscrepancy.values.sum / areasWithMeanDiscrepancy.size
      } else {
        Double.NaN
      }
    )
  }

  def discrepancy[T](
      node: SimpleNodeManager[T],
      other: SimpleNodeManager[T]
  ): Double = {
    if (
      node.getOption(Sensors.model).isEmpty || other
        .getOption(Sensors.model)
        .isEmpty
    ) Double.PositiveInfinity
    else {
      val nodeModel = node.get[py.Dynamic](Sensors.model)
      val otherModel = other.get[py.Dynamic](Sensors.model)
      utils
        .discrepancy(nodeModel, otherModel)
        .as[Double]
    }
  }

}
