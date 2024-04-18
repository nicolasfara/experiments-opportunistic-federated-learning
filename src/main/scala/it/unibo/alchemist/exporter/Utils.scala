package it.unibo.alchemist.exporter

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Node, Position}

import scala.jdk.CollectionConverters.IteratorHasAsScala

object Utils {
  implicit class EnvironmentOps[P <: Position[P], T](val environment: Environment[T, _]) extends AnyVal {
    def getNodesAsScala: List[Node[T]] = environment.getNodes.iterator().asScala.toList
  }

  implicit def stringToMolecule(s: String): SimpleMolecule = new SimpleMolecule(s)
}
