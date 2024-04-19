package it.unibo.alchemist.exporter

import it.unibo.alchemist.boundary.extractors.AbstractDoubleExporter
import it.unibo.alchemist.model.{Actionable, Environment, Time}

import java.{lang, util}
import scala.jdk.CollectionConverters.SeqHasAsJava
import it.unibo.Utils._
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
class AreaDiscrepancyExporter(areas: Int) extends AbstractDoubleExporter {

  override def getColumnNames: util.List[String] = (1 to areas).map(i => s"AreaDiscrepancy$i").toList.asJava

  override def extractData[T](environment: Environment[T, _], actionable: Actionable[T], time: Time, l: Long): util.Map[String, lang.Double] = {
    val nodes = environment.getNodesAsScala.map(new SimpleNodeManager[T](_))
    val leaders = nodes.filter(node => node.has("aggregators"))
      .map(node => node.get("agg"))
    ???
  }
}
