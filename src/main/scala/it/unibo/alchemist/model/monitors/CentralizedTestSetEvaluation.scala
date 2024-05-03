package it.unibo.alchemist.model.monitors

import it.unibo.alchemist.exporter.TestDataExporter
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Position, Time}
import it.unibo.scafi.Sensors
import it.unibo.scafi.interop.PythonModules.utils
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}
import me.shadaj.scalapy.py

class CentralizedTestSetEvaluation[P <: Position[P]](seed: Double, epochs: Int, areas: Int, dataShuffle: Boolean)
    extends TestSetEvaluation[P](seed, epochs, areas, dataShuffle) {

  override def finished(environment: Environment[Any, P], time: Time, step: Long): Unit = {
    println("Starting evaluation...")
    val weights =
      nodes(environment).head.getConcentration(new SimpleMolecule(Sensors.model)).asInstanceOf[py.Dynamic]
    val data = utils.get_dataset(Seq.empty[Int].toPythonProxy, false, false)
    val accuracy = evaluate(weights, data)
    TestDataExporter.CSVExport(
      List(accuracy),
      s"data-test-baseline/test_accuracy_seed-${seed}_epochs-${epochs}" +
        s"_areas-${areas}_batchSize-${batch_size}_dataShuffle-${dataShuffle}"
    )
    val gc = py.module("gc")
    try {
      val pythonObjects = py"list($gc.get_objects())".as[Seq[py.Dynamic]]
      for (elem <- pythonObjects) {
        py"del $elem"
      }
      gc.collect()
    } catch {
      case e: Exception => println(e)
    }
  }

}
