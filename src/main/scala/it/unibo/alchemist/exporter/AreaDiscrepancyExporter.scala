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
class AreaDiscrepancyExporter(areas: Int) extends AbstractDoubleExporter {

  override def getColumnNames: util.List[String] = (0 until areas).map(i => s"AreaDiscrepancy$i").toList.asJava

  override def extractData[T](environment: Environment[T, _], actionable: Actionable[T], time: Time, l: Long): util.Map[String, lang.Double] = {
    val nodes = environment.getNodesAsScala.map(_.manager)
    val leaders = nodes.filter(node => node.getOrElse(Sensors.isAggregator, false))
    val leadersWithLabels =
      leaders.map(leader => leader -> leader.getOrElse(Sensors.labels, Set.empty[Int]))
        .groupMap(_._2) { case (leader, labels) => computeAverageDiscrepancy(leader) }
        .map { case (labels, discrepancies) => labels -> discrepancies.sum / discrepancies.size }
        .toList
        .sortBy(_._1.hashCode())

    leadersWithLabels.zipWithIndex
      .map({ case ((_, discrepancy), i) => s"AreaDiscrepancy$i" -> discrepancy.asInstanceOf[java.lang.Double] })
      .toMap
      .asJava
  }

  def computeAverageDiscrepancy[T](leader: SimpleNodeManager[T]): Double = {
    val leaderModel = leader.get[py.Dynamic](Sensors.model)
    val areaModels = leader.get[Set[py.Dynamic]](Sensors.models)
    val discrepancies = areaModels.map(nodeModel =>
      utils.discrepancy(leaderModel, nodeModel).as[Double])
    discrepancies.sum / discrepancies.size
  }
}
