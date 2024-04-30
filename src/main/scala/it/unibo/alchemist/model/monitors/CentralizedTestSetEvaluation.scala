package it.unibo.alchemist.model.monitors

import it.unibo.alchemist.exporter.TestDataExporter
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Position, Time}
import it.unibo.scafi.Sensors
import it.unibo.scafi.interop.PythonModules.utils
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py

class CentralizedTestSetEvaluation[P <: Position[P]] extends TestSetEvaluation[P] {

  override def finished(environment: Environment[Any, P], time: Time, step: Long): Unit = {
    println("Starting evaluation...")
    val weights =
      nodes(environment).head.getConcentration(new SimpleMolecule(Sensors.model)).asInstanceOf[py.Dynamic]
    val data = utils.get_dataset(Seq.empty[Int].toPythonProxy, false)
    val accuracy = evaluate(weights, data)
    TestDataExporter.CSVExport(List(accuracy), "data-baseline/test-accuracy")
  }

}
