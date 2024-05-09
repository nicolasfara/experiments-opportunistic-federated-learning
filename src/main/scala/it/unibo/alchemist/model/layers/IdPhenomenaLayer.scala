package it.unibo.alchemist.model.layers

import it.unibo.alchemist.model._
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.scafi.interop.PythonModules
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

class IdPhenomenaLayer[P <: Position[P]](
    environment: Environment[_, P],
    phenomenaMolecule: String
) extends Layer[Double, P] {

  override def getValue(p: P): Double = {
    val layer = environment
      .getLayer(new SimpleMolecule(phenomenaMolecule))
      .get()
    val data = layer.asInstanceOf[PhenomenaDistribution[P]].getValue(p)
    data.areaId.toDouble / layer.asInstanceOf[PhenomenaDistribution[P]].areas
  }

}
