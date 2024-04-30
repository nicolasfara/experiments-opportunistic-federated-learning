package it.unibo.alchemist.model

import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.scafi.Sensors
import me.shadaj.scalapy.py
import scala.jdk.CollectionConverters.IteratorHasAsScala
import it.unibo.scafi.interop.PythonModules.utils
import me.shadaj.scalapy.py.PyQuote

class TestSetEvaluation[P <: Position[P]] extends OutputMonitor[Any, P] {

  private val batch_size = 64
  private val seed = 1

  override def finished(environment: Environment[Any, P], time: Time, step: Long): Unit = {
    println("Starting evaluation...")
    val layer = environment.getLayer(new SimpleMolecule(Sensors.testsetPhenomena)).get()
    val accuracies =
      nodes(environment)
      .map(node => {
        val weights = node.getConcentration(new SimpleMolecule(Sensors.model)).asInstanceOf[py.Dynamic]
        val data = layer.getValue(environment.getPosition(node)).asInstanceOf[py.Dynamic]
        (weights, data)
      })
      .map { case (w, d) => evaluate(w, d) }
    accuracies.foreach(println(_))
  }

  private def nodes(environment: Environment[Any, P]): List[Node[Any]] =
    environment.getNodes.iterator().asScala.toList

  private def evaluate(weights: py.Dynamic, data: py.Dynamic): Double = {
    val model = utils.nn_from_weights(weights)
    val result = utils.evaluate(model, data, batch_size, seed)
    val accuracy = py"$result[0]".as[Double]
    accuracy
  }

}
