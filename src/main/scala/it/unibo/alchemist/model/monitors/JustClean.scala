package it.unibo.alchemist.model.monitors

import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.alchemist.model.layers.PhenomenaDistribution
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Node, Position, Time}
import it.unibo.scafi.Sensors
import it.unibo.scafi.interop.PythonModules.utils
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.PyQuote

import scala.jdk.CollectionConverters.IteratorHasAsScala

class JustClean[P <: Position[P]](seed: Double) extends OutputMonitor[Any, P] {

  protected val batch_size = 64

  override def finished(
      environment: Environment[Any, P],
      time: Time,
      step: Long
  ): Unit = {
    cleanPythonObjects(environment)
  }

  def cleanPythonObjects(environment: Environment[_, P]): Unit = {
    val gc = py.module("gc")

    try {
      environment
        .getLayer(new SimpleMolecule(Sensors.phenomena))
        .ifPresent(layer => layer.asInstanceOf[PhenomenaDistribution[P]].cleanAll())
      val nodes = environment.getNodes.iterator().asScala.toList
      nodes.foreach { node =>
        node.getConcentration(new SimpleMolecule(Sensors.model)).asInstanceOf[py.Dynamic].del()
        node.getConcentration(new SimpleMolecule(Sensors.sharedModel)).asInstanceOf[py.Dynamic].del()

      }
      gc.collect()
      Runtime.getRuntime.gc()
    } catch {
      case e: Throwable => println(e)
    }
  }
}
