package it.unibo.alchemist.model.linkingrules

import it.unibo.alchemist.model.Environment
import it.unibo.alchemist.model.Neighborhood
import it.unibo.alchemist.model.Node
import it.unibo.alchemist.model.Position
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.neighborhoods.Neighborhoods

class ConnectWithinDistanceAndPhenomena<T, P : Position<P>>(
    radius: Double,
    private val nearestPhenomenaConnections: Int,
) : ConnectWithinDistance<T, P>(radius) {
    private val phenomenaMolecule = SimpleMolecule("Phenomena")
    override fun computeNeighborhood(center: Node<T>, environment: Environment<T, P>): Neighborhood<T> {
        val phenomenaNodes = environment.nodes.filter { it.contains(phenomenaMolecule) }
        return when {
            center.contains(phenomenaMolecule) -> {
                val simpleNodes = environment.nodes.filter { !it.contains(phenomenaMolecule) }
                val associations = simpleNodes.associateWith { n ->
                    phenomenaNodes.map { p -> p to environment.getDistanceBetweenNodes(p, n) }
                }.mapValues { (_, values) -> values.minBy { it.second } }
                    .filter { (_, value) -> value.first == center }
                    .keys
                Neighborhoods.make(environment, center, associations)
            }
            else -> {
                val nearestPhenomena = phenomenaNodes.asSequence()
                    .map { it to environment.getDistanceBetweenNodes(center, it) }
                    .sortedBy { it.second }
                    .take(nearestPhenomenaConnections)
                    .map { it.first }
                super.computeNeighborhood(center, environment).apply {
                    // removeAll { it.contains(SimpleMolecule("Phenomena")) }
                    nearestPhenomena.forEach { this.add(it) }
                }
            }
        }
    }
}