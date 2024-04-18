package it.unibo.alchemist.exporter

import it.unibo.alchemist.boundary.extractors.AbstractDoubleExporter
import it.unibo.alchemist.model.{Actionable, Environment, Time}

import java.{lang, util}
import scala.jdk.CollectionConverters.SeqHasAsJava
import Utils._
class AreaDiscrepancyExporter(areas: Int) extends AbstractDoubleExporter {

  override def getColumnNames: util.List[String] = (1 to areas).map(i => s"AreaDiscrepancy$i").toList.asJava

  override def extractData[T](environment: Environment[T, _], actionable: Actionable[T], time: Time, l: Long): util.Map[String, lang.Double] = ???
}
