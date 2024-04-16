package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import interop.PythonModules._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.PyQuote

class OpportunisticFederatedLearning
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with FieldUtils
    with BuildingBlocks {

  private def computeMetric(myModel: py.Dynamic, otherModule: py.Dynamic): Double = {
    val discrepancy = utils.discrepancy(myModel.state_dict(), otherModule.state_dict())
    py"$discrepancy".as[Double]
  }

  private val localModel = utils.cnn_factory() // TODO - implement
  private val epochs = 2
  private val discrepancyThreshold = 1.0 // TODO

  override def main(): Any = {

    val data = utils.get_dataset(mid()) // TODO - implement
    val aggregators = S(
      discrepancyThreshold,
      metric = () => discrepancyMetric(localModel, nbr(localModel))
    )
    42
//    val model = FooModel(mid())
//    val models = excludingSelf.reifyField(nbr(model))
//    // first metric
//    val evaluations = models.map { case(id, model) => id -> model.eval(data) } // T
//    val neighEvals = excludingSelf.reifyField(nbr(evaluations)).map { case(id, evals) => id -> evals(id) } // T - 1
//    models.keys.map(id => id -> (evaluations(id) + neighEvals(id)) / 2).toMap
//    // second metric (discrepancy)
//    val discrepancies = models.map { case(id, model) => id -> model.diff(model) } // T
  }

  private def discrepancyMetric(
      myModel: py.Dynamic,
      otherModule: py.Dynamic
  ): Double =
    utils.discrepancy(myModel.state_dict(), otherModule.state_dict())

  private def evalModel(myModel: py.Dynamic): Double = ??? // TODO - implement

  private def accuracyBasedMetric(model: py.Dynamic): Double = {
    val models = includingSelf.reifyField(nbr(model))
    val evaluations = models.map { case (id, model) => id -> evalModel(model) }
    val neighEvals = excludingSelf.reifyField(nbr(evaluations))
    (neighEvals(mid)(nbr(mid)) + neighEvals(nbr(mid))(mid)) / 2
  }
}
