package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import interop.PythonModules._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}
import Sensors._
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist

/** Metriche loss media per ogni area / set di etichette (nico) --
  * validation/test loss [X] loss globale (nico) accuracy media per ogni area
  * (nico) (di validation) accuracy globale (dom) divergenza (all'interno
  * dell'area) -- gianlu corretteza della aree (i nodi che hanno lo stesso
  * dataset sono nella stessa area) -- nico [X] convergenza (specifico sul
  * movimento) -- io accuracy + loss su test -- dom
  *
  * algoritmo fedarato centrizzato (baseline) -- dom aggiungi validation loss
  * per ogni nodo (davide) posizionamento del dato in base alla posizione
  * spaziale (idea: fare una griglia di nodi che non eseguono il programma ma
  * servono solo per posizionare i dati e poi usi 1-nn search per trovare i
  * dati) -- nico usare più aree (io) usare aree fuzzy (k=2) -- gianlu movimento
  * di un nodo -- gianlu con più nodi (???) -- gianlu
  */
class OpportunisticFederatedLearning
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with FieldUtils
    with BuildingBlocks {

  private lazy val localModel = utils.cnn_loader(seed())
  private lazy val data = utils.get_dataset(indexes())
  private lazy val (trainData, valData) = splitDataset()
  private val boundedInt: Builtins.Bounded[ID] = Builtins.Bounded.of_i
  private val boundedDouble: Builtins.Bounded[Double] = Builtins.Bounded.of_d
  implicit val boundedOfTuple: ScafiIncarnationForAlchemist.Builtins.Bounded[(Double, Int, Double, Double)] = Builtins.Bounded.tupleBounded4(boundedDouble, boundedInt, boundedDouble, boundedDouble)
  private def actualMetric: (py.Dynamic) => () => Double = (model) => {
    val models = includingSelf.reifyField(nbr(model))
    val evaluations = models.map { case (id, model) =>
      id -> evalModel(model, trainData)._2
    }
    val neighEvals = includingSelf.reifyField(nbr(evaluations))
    () => accuracyBasedMetric(neighEvals)
  }
  private val epochs = 2
  private val batch_size = 64
  private val every = 5
  private lazy val threshold = sense[Double](lossThreshold)

  override def main(): Any = {
    rep((localModel, localModel, 1)) { case (local, global, tick) =>
      val metric = actualMetric(global)
      val isAggregator = S(
        threshold,
        metric = metric
      )
      val (evolvedModel, trainLoss) = localTraining(local, trainData)
      val (validationAccuracy, validationLoss) =
        evalModel(evolvedModel, valData)
      val neighbourhoodMetric = excludingSelf.reifyField(metric())
      val potential = classicGradient(isAggregator, metric)
      val sender = G_along(potential, metric, mid(), (_: ID) => nbr(mid()))
      val leader = broadcast(isAggregator, mid(), metric)
      val info = CWithSenderField[List[py.Dynamic]](
        _ ++ _,
        List(evolvedModel),
        List.empty,
        sender
      )
      val aggregatedModel = averageWeights(info, List.fill(info.length)(1.0))
      val sharedModel = broadcast(isAggregator, aggregatedModel, metric)
      if (isAggregator) { snapshot(sharedModel, mid(), tick) }
      // Actuations
      node.put(Sensors.leaderId, leader)
      node.put(Sensors.model, local)
      if (isAggregator) { node.put(models, info) }
      node.put(Sensors.potential, potential)
      node.put(Sensors.modelsCount, info.length)
      node.put(Sensors.neighbourhoodMetric, neighbourhoodMetric)
      node.put(Sensors.isAggregator, isAggregator)
      node.put(Sensors.trainLoss, trainLoss)
      node.put(Sensors.validationLoss, validationLoss)
      node.put(Sensors.validationAccuracy, validationAccuracy)
      node.put(Sensors.parent, findParentWithMetric(potential, metric))
      node.put(Sensors.sender, sender)
      mux(impulsesEvery(tick)) {
        (
          averageWeights(List(evolvedModel, sharedModel), List(0.1, 0.9)),
          averageWeights(List(evolvedModel, sharedModel), List(0.1, 0.9)),
          tick + 1
        )
      } {
        (evolvedModel,
          averageWeights(List(evolvedModel, sharedModel), List(0.9, 0.1)),
          tick + 1)
      }
    }
  }

  private def localTraining(
      model: py.Dynamic,
      trainData: py.Dynamic
  ): (py.Dynamic, Double) = {
    val result = utils.local_training(model, epochs, trainData, batch_size, seed())
    val newWeights = py"$result[0]"
    val trainLoss = py"$result[1]".as[Double]
    val freshNN = utils.cnn_loader(seed())
    freshNN.load_state_dict(newWeights)
    (freshNN, trainLoss)
  }

  private def discrepancyMetric(
      myModel: py.Dynamic,
      otherModule: py.Dynamic
  ): Double = {
    val discrepancy =
      utils.discrepancy(myModel.state_dict(), otherModule.state_dict())
    py"$discrepancy".as[Double]
  }

  private def sample(model: py.Dynamic): py.Dynamic =
    model.state_dict()

  private def averageWeights(models: List[py.Dynamic], weights: List[Double]): py.Dynamic = {
    val averageWeights =
      utils.average_weights(models.map(sample).toPythonProxy, weights.toPythonProxy)
    val freshNN = utils.cnn_loader(seed())
    freshNN.load_state_dict(averageWeights)
    freshNN
  }

  private def evalModel(
      myModel: py.Dynamic,
      validationData: py.Dynamic
  ): (Double, Double) = {
    val result = utils.evaluate(myModel, validationData, batch_size, seed())
    val accuracy = py"$result[0]".as[Double]
    val loss = py"$result[1]".as[Double]
    (accuracy, loss)
  }

  private def accuracyBasedMetric(
      neighEvals: Map[ID, Map[ID, Double]]
  ): Double = {
    def directLinkMeToNeigh(): Double =
      neighEvals
        .getOrElse(mid(), Map.empty)
        .getOrElse(nbr(mid()), Double.PositiveInfinity)
    def directLinkNeighToMe(): Double =
      neighEvals
        .getOrElse(nbr(mid()), Map.empty)
        .getOrElse(mid(), Double.PositiveInfinity)
    (directLinkMeToNeigh + directLinkNeighToMe)
  }

  private def snapshot(model: py.Dynamic, id: Int, tick: Int): Unit = {
    torch.save(
      model.state_dict(),
      s"networks/aggregator-$id-time-$tick"
    )
  }

  private def seed(): Int = node.get[Double](Sensors.seed).toInt

  private def indexes() = node.get[List[Int]](Sensors.data).toPythonProxy

  private def impulsesEvery(time: Int): Boolean = time % every == 0

  private def splitDataset(): (py.Dynamic, py.Dynamic) = {
    val datasets = utils.train_val_split(data, seed())
    val trainData = py"$datasets[0]"
    val valData = py"$datasets[1]"
    (trainData, valData)
  }

  def CWithMetric[V](
      potential: Double,
      acc: (V, V) => V,
      local: V,
      Null: V,
      metric: () => Double
  ): V =
    share(local) { case (_, query) =>
      acc(
        local,
        foldhood(Null)(acc) {
          mux(nbr(findParentWithMetric(potential, metric)) == mid()) {
            query()
          } { nbr(Null) }
        }
      )
    }

  def CWithSenderField[V](
   acc: (V, V) => V,
   local: V,
   Null: V,
   parentField: ID
 ): V = share(local) { case (_, query) =>
    acc(
      local,
      foldhoodPlus(Null)(acc) {
        mux(nbr(parentField) == mid()) {
          query()
        } { nbr(Null) }
      }
    )
  }

  def findParentWithMetric(potential: Double, metric: () => Double): ID = {
    val others =
      excludingSelf.reifyField((potential - nbr(potential)).similarTo(metric()))
    mux(potential.similarTo(0)) {
      Int.MaxValue
    } {
      others.find(_._2).map(_._1).getOrElse(Int.MaxValue)
    }
  }

  implicit class RichDouble(d: Double) {
    def similarTo(other: Double, precision: Double = 0.01): Boolean =
      d < (other + precision) && other < (d + precision)
  }

  override def flexGradient(epsilon: Double = DEFAULT_FLEX_CHANGE_TOLERANCE_EPSILON,
                   delta: Double = DEFAULT_FLEX_DELTA,
                   communicationRadius: Double = DEFAULT_FLEX_DELTA)
                  (source: Boolean,
                   metric: Metric = nbrRange
                  ): Double =
    share(Double.PositiveInfinity){ case (local, query) =>
      import Builtins.Bounded._ // for min/maximizing over tuplesù
      def distance = Math.max(metric(), delta * communicationRadius)
      val maxLocalSlope: (Double, ID, Double, Double) =
        maxHood { ((local - query()) / distance, nbr{ mid }, query(), metric()) }
      val constraint = minHoodPlus{ (query() + distance) }
      mux(source){ 0.0 }{
        if(Math.max(communicationRadius, 2 * constraint) < local) {
          constraint
        } else if(maxLocalSlope._1 > 1 + epsilon) {
          maxLocalSlope._3 + (1 + epsilon) * Math.max(delta * communicationRadius, maxLocalSlope._4)
        } else if(maxLocalSlope._1 < 1 - epsilon){
          maxLocalSlope._3 + (1 - epsilon) * Math.max(delta * communicationRadius, maxLocalSlope._4)
        } else {
          local
        }
      }
    }
}
