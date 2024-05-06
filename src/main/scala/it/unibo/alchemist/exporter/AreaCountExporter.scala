package it.unibo.alchemist.exporter

import it.unibo.alchemist.boundary.extractors.AbstractDoubleExporter
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Actionable, Environment, Time}

import java.{lang, util}
import scala.jdk.CollectionConverters._
import it.unibo.Utils._
import it.unibo.scafi.Sensors
class AreaCountExporter extends AbstractDoubleExporter {

  override def getColumnNames: util.List[String] = util.List.of("AreaCount")

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: Time,
      l: Long
  ): util.Map[String, lang.Double] = {
    val nodes = environment.getNodesAsScala
    val areas = nodes
      .filter(node => node.getConcentration(Sensors.isAggregator).asInstanceOf[Boolean])
      .map(node => node.getConcentration(Sensors.leaderId))
      .map(_.asInstanceOf[Int])
      .distinct
      .length
    util.Map.of("AreaCount", areas)
  }
}
