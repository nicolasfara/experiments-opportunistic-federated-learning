package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import interop.PythonModules._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}
import Sensors._
/**
 * Metriche
 *  loss media per ogni area / set di etichette (nico) -- validation/test loss [X]
 *  loss globale (nico)
 *  accuracy media per ogni area (nico) (di validation)
 *  accuracy globale (dom)
 *  divergenza (all'interno dell'area) -- gianlu
 *  corretteza della aree (i nodi che hanno lo stesso dataset sono nella stessa area) -- nico [X]
 *  convergenza
 *  (specifico sul movimento) -- io
 *  accuracy + loss su test -- dom
 *
 * algoritmo fedarato centrizzato (baseline) -- dom
 * aggiungi validation loss per ogni nodo (davide)
 * posizionamento del dato in base alla posizione spaziale (idea: fare una griglia di nodi che non eseguono il programma ma servono solo per posizionare i dati e poi usi 1-nn search per trovare i dati) -- nico
 * usare più aree (io)
 * usare aree fuzzy (k=2) -- gianlu
 * movimento di un nodo -- gianlu
 * con più nodi (???) -- gianlu
 */
class OpportunisticFederatedLearning
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with FieldUtils
    with BuildingBlocks {

  private lazy val localModel = utils.cnn_loader(seed())
  private lazy val data = utils.get_dataset(indexes())
  private def actualMetric: (py.Dynamic) => () => Double = (model) => {
          val models = includingSelf.reifyField(nbr(model))
          val evaluations = models.map { case (id, model) => id -> evalModel(model, data)._2 }
          val neighEvals = includingSelf.reifyField(nbr(evaluations))
          () => accuracyBasedMetric(neighEvals)
    }
  private val epochs = 2
  private val batch_size = 64
  private val every = 2
  private lazy val threshold = sense[Double](lossThreshold)

  override def main(): Any = {
    rep((localModel, 1)) { case (model, tick) =>
      val metric = actualMetric(model)
      val isAggregator = S(
        threshold,
        metric = metric
      )
      val (trainData, valData) = splitDataset()
      val (evolvedModel, trainLoss) = localTraining(model, trainData)
      val (validationAccuracy, validationLoss) = evalModel(evolvedModel, valData)
      val neighbourhoodMetric = excludingSelf.reifyField(metric())
      val potential = classicGradient(isAggregator, metric)
      val leader = broadcast(isAggregator, mid(), metric)
      val info = CWithMetric[List[py.Dynamic]](
        potential,
        _ ++ _,
        List(evolvedModel),
        List.empty,
        metric
      )
      node.put("potential", potential)
      val aggregatedModel = averageWeights(info.map(sample))
      val sharedModel = broadcast(isAggregator, aggregatedModel, metric)
      if (isAggregator) { snapshot(sharedModel, mid(), tick) }
      // Actuations
      node.put(Sensors.leaderId, leader)
      node.put(Sensors.model, sharedModel)
      if(isAggregator) { node.put(models, info) }
      node.put(Sensors.neighbourhoodMetric, neighbourhoodMetric)
      node.put(Sensors.isAggregator, isAggregator)
      node.put(Sensors.trainLoss, trainLoss)
      node.put(Sensors.validationLoss, validationLoss)
      node.put(Sensors.validationAccuracy, validationAccuracy)
      node.put("parent", findParentWithMetric(potential, metric))
      mux(impulsesEvery(tick)) {
        (
          sharedModel,
          tick + 1
        )
      } {
        (evolvedModel, tick + 1)
      }
    }
  }

  private def localTraining(
      model: py.Dynamic,
      trainData: py.Dynamic
  ): (py.Dynamic, Double) = {
    val result = utils.local_training(model, epochs, trainData, batch_size)
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

  private def averageWeights(models: List[py.Dynamic]): py.Dynamic = {
    val averageWeights =
      utils.average_weights(models.toPythonProxy)
    val freshNN = utils.cnn_loader(seed())
    freshNN.load_state_dict(averageWeights)
    freshNN
  }

  private def evalModel(myModel: py.Dynamic, validationData: py.Dynamic): (Double, Double) = {
    val result = utils.evaluate(myModel, validationData, batch_size)
    val accuracy = py"$result[0]".as[Double]
    val loss = py"$result[1]".as[Double]
    (accuracy, loss)
  }

  private def accuracyBasedMetric(neighEvals: Map[ID, Map[ID, Double]]): Double = {
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
    val datasets = utils.train_val_split(data)
    val trainData = py"$datasets[0]"
    val valData = py"$datasets[1]"
    (trainData, valData)
  }

  def CWithMetric[V](potential: Double, acc: (V, V) => V, local: V, Null: V, metric: () => Double): V =
    rep(local) { case (query) =>
      acc(local, foldhood(Null)(acc) {
        mux(nbr(findParentWithMetric(potential, metric) == mid())) { nbr(query) } { nbr(Null) }
      })
    }

  def findParentWithMetric(potential: Double, metric: () => Double): ID = {
    val others = excludingSelf.reifyField((potential - nbr(potential)).similarTo(metric()))
    node.put("parents", others)
    mux(potential == 0.0) {
      Int.MaxValue
    } {
      others.find(_._2).map(_._1).getOrElse(Int.MaxValue)
    }
  }

  implicit class RichDouble(d: Double) {
    def similarTo(other: Double, precision: Double = 0.01): Boolean =
      smaller(d, other + precision) && smaller(other, d + precision)
  }

  private def smaller(a: Double, b: Double): Boolean =
    a < b
}