package it.unibo.alchemist.exporter

import it.unibo.alchemist.boundary.extractors.AbstractDoubleExporter
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Actionable, Environment, Time}

import java.{lang, util}
import scala.jdk.CollectionConverters._

class AreaCountExporter extends AbstractDoubleExporter {

  override def getColumnNames: util.List[String] = util.List.of("AreaCount")

  override def extractData[T](environment: Environment[T, _], actionable: Actionable[T], time: Time, l: Long): util.Map[String, lang.Double] = {
    val nodes = environment.getNodes.iterator().asScala.toList
    val areas = nodes.map(node => node.getConcentration(new SimpleMolecule("leader"))).map(_.asInstanceOf[Int]).distinct.length
    util.Map.of("AreaCount", areas)
  }
}
