package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

class OpportunisticFederatedLearning
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with FieldUtils {

  lazy val data = randomGen.nextDouble()
  trait Model {
    def eval(data: Double): Double
    def diff(model: Model): Double
  }

  case class FooModel(id: ID) extends Model {
    override def eval(data: Double): Double = data + id
    override def diff(model: Model): Double = id
  }

  override def main(): Any = {
    42
//    val model = FooModel(mid())
//    val models =  excludingSelf.reifyField(nbr(model))
//    // first metric
//    val evaluations = models.map { case(id, model) => id -> model.eval(data) } // T
//    val neighEvals = excludingSelf.reifyField(nbr(evaluations)).map { case(id, evals) => id -> evals(id) } // T - 1
//    models.keys.map(id => id -> (evaluations(id) + neighEvals(id)) / 2).toMap
//    // second metric (discrepancy)
//    val discrepancies = models.map { case(id, model) => id -> model.diff(model) } // T
  }
}
