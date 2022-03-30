package maia.ml.evaluation

import maia.ml.dataset.DataBatch
import maia.ml.dataset.DataRow
import maia.ml.dataset.DataStream
import maia.ml.dataset.WithColumns
import maia.ml.learner.Learner
import maia.ml.learner.type.LearnerType
import maia.ml.learner.type.intersectionOf
import maia.util.forEachReversed

/**
 * Base-class for aggregate evaluators which combine multiple metrics under a
 * single group-name. The metric type is a map from each evaluator's name to its
 * respective metric.
 *
 * @param name
 *          The name to give the group of metrics.
 * @param subEvaluators
 *          The evaluators to group.
 *
 * @param D
 *          The type of data-set this group of evaluators can work with.
 * @param L
 *          The type of learner this group can work with.
 * @param M
 *          The type of metric the evaluators in the group produce.
 * @param E
 *          The type of the evaluators.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
sealed class GroupEvaluator<in D: DataStream<*>, in L: Learner<D>, out M, out E: Evaluator<D, L, M>>(
    override val name : String,
    protected vararg val subEvaluators : E
) : Evaluator<D, L, M> {

    companion object {

        /**
         * Creates a group of batch-evaluators.
         *
         * @param name
         *          The name to give the group of metrics.
         * @param evaluators
         *          The batch-evaluators to group.
         *
         * @param L
         *          The type of learner this group can work with.
         * @param M
         *          The type of metric the evaluators produce.
         *
         * @return
         *          The group evaluator.
         */
        fun <L: Learner<DataBatch<*>>, M> create(name: String, vararg evaluators: BatchEvaluator<L, M>): GroupBatchEvaluator<L, M> {
            return GroupBatchEvaluator(name, *evaluators)
        }

        /**
         * Creates a group of stream-evaluators.
         *
         * @param name
         *          The name to give the group of metrics.
         * @param evaluators
         *          The stream-evaluators to group.
         *
         * @param L
         *          The type of learner this group can work with.
         * @param M
         *          The type of metric the evaluators produce.
         *
         * @return
         *          The group evaluator.
         */
        fun <L: Learner<DataStream<*>>, M> create(name: String, vararg evaluators: StreamEvaluator<L, M>): GroupStreamEvaluator<L, M> {
            return GroupStreamEvaluator(name, *evaluators)
        }
    }

    override val type : LearnerType = intersectionOf(*subEvaluators.map { it.type }.toTypedArray())

    override val metric : Metric.Group<M>
        get() {
            return Metric.Group(
                LinkedHashMap<String, Metric<M>>(subEvaluators.size).apply {
                    subEvaluators.forEach {
                        this[it.name] = it.metric
                    }
                }
            )
        }

    override fun preEvaluate(vararg learners: L) =
        subEvaluators.forEach { it.preEvaluate(*learners) }

    override fun postEvaluate(vararg learners: L) =
        subEvaluators.forEachReversed { it.postEvaluate(*learners) }

    override fun preInitialise(learner : L, learnerIndex: Int, headers: WithColumns) =
        subEvaluators.forEach { it.preInitialise(learner, learnerIndex, headers) }

    override fun postInitialise(learner : L, learnerIndex: Int, headers: WithColumns) =
        subEvaluators.forEachReversed { it.postInitialise(learner, learnerIndex, headers) }

    override fun preTrain(learner : L, learnerIndex: Int, dataset : D) =
        subEvaluators.forEach { it.preTrain(learner, learnerIndex, dataset) }

    override fun postTrain(learner : L, learnerIndex: Int, dataset : D) =
        subEvaluators.forEachReversed { it.postTrain(learner, learnerIndex, dataset) }

    override fun prePredict(learner : L, learnerIndex: Int, row : DataRow) =
        subEvaluators.forEach { it.prePredict(learner, learnerIndex, row) }

    override fun postPredict(learner : L, learnerIndex: Int, row : DataRow, prediction : DataRow) =
        subEvaluators.forEachReversed { it.postPredict(learner, learnerIndex, row, prediction) }

}

/**
 * Aggregate evaluator which combine multiple batch-evaluators under a
 * single group-name. The metric type is a map from each evaluator's name to its
 * respective metric.
 *
 * @param name
 *          The name to give the group of metrics.
 * @param subEvaluators
 *          The evaluators to group.
 *
 * @param L
 *          The type of learner this group can work with.
 * @param M
 *          The type of metric the evaluators produce.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
class GroupBatchEvaluator<in L: Learner<DataBatch<*>>, out M>(
    name: String,
    vararg subEvaluators: BatchEvaluator<L, M>
) : GroupEvaluator<DataBatch<*>, L, M, BatchEvaluator<L, M>>(
    name,
    *subEvaluators
), BatchEvaluator<L, M>

/**
 * Aggregate evaluator which combine multiple stream-evaluators under a
 * single group-name. The metric type is a map from each evaluator's name to its
 * respective metric.
 *
 * @param name
 *          The name to give the group of metrics.
 * @param subEvaluators
 *          The evaluators to group.
 *
 * @param L
 *          The type of learner this group can work with.
 * @param M
 *          The type of metric the evaluators produce.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
class GroupStreamEvaluator<in L: Learner<DataStream<*>>, out M>(
    name: String,
    vararg subEvaluators: StreamEvaluator<L, M>
) : GroupEvaluator<DataStream<*>, L, M, StreamEvaluator<L, M>>(
    name,
    *subEvaluators
), StreamEvaluator<L, M> {

    override fun perRowCallback(
        learner : L,
        learnerIndex : Int,
        dataset : DataStream<*>
    ) : EvaluationHarnessCallback? {
        val subCallBacks = ArrayList<EvaluationHarnessCallback>(subEvaluators.size)
        subEvaluators.forEach { evaluator ->
            evaluator.perRowCallback(learner, learnerIndex, dataset)?.let { subCallBacks.add(it) }
        }
        return if (subEvaluators.isEmpty())
            null
        else
            { row -> subCallBacks.forEach { it(row) } }
    }
}
