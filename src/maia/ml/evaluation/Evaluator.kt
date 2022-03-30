package maia.ml.evaluation

import maia.ml.dataset.DataBatch
import maia.ml.dataset.DataRow
import maia.ml.dataset.DataStream
import maia.ml.dataset.WithColumns
import maia.ml.learner.Learner
import maia.ml.learner.type.LearnerType

/**
 * Base interface for classes which can calculate a metric over the course of an
 * evaluation.
 *
 * @param D
 *          The type of dataset the evaluation can be performed over.
 * @param L
 *          The type of learner that the evaluator needs to measure.
 * @param M
 *          The type of metric that the evaluator measures.
 */
sealed interface Evaluator<in D: DataStream<*>, in L: Learner<D>, out M> {
    /** The name of the metric this evaluator measures. */
    val name: String

    /** The type of (initialised) learner this evaluator can study. */
    val type: LearnerType

    /** The current value of the metric. */
    val metric: Metric<M>

    /**
     * Called at the beginning of a new evaluation.
     *
     * @param learners
     *          The learners being evaluated.
     */
    fun preEvaluate(vararg learners: L)

    /**
     * Called at the end of an evaluation.
     *
     * @param learners
     *          The learners being evaluated.
     */
    fun postEvaluate(vararg learners: L)

    /**
     * Called before a learner is initialised.
     *
     * @param learner
     *          The learner being initialised.
     * @param learnerIndex
     *          The index of the learner in the evaluation.
     * @param headers
     *          The headers being used to initialise the learner.
     */
    fun preInitialise(learner: L, learnerIndex: Int, headers: WithColumns)

    /**
     * Called after a learner is initialised.
     *
     * @param learner
     *          The learner being initialised.
     * @param learnerIndex
     *          The index of the learner in the evaluation.
     * @param headers
     *          The headers being used to initialise the learner.
     */
    fun postInitialise(learner: L, learnerIndex: Int, headers: WithColumns)

    /**
     * Called before a learner is trained on a dataset.
     *
     * @param learner
     *          The learner being trained.
     * @param learnerIndex
     *          The index of the learner in the evaluation.
     * @param dataset
     *          The dataset being used to train the learner.
     */
    fun preTrain(learner : L, learnerIndex: Int, dataset: D)

    /**
     * Called after a learner is trained on a dataset.
     *
     * @param learner
     *          The learner being trained.
     * @param learnerIndex
     *          The index of the learner in the evaluation.
     * @param dataset
     *          The dataset being used to train the learner.
     */
    fun postTrain(learner : L, learnerIndex: Int, dataset : D)

    /**
     * Called before a learner performs a prediction on a data-row.
     *
     * @param learner
     *          The learner being used for predicting.
     * @param learnerIndex
     *          The index of the learner in the evaluation.
     * @param row
     *          The data-row being predicted against.
     */
    fun prePredict(learner: L, learnerIndex: Int, row: DataRow)

    /**
     * Called after a learner performs a prediction on a data-row.
     *
     * @param learner
     *          The learner being used for predicting.
     * @param learnerIndex
     *          The index of the learner in the evaluation.
     * @param row
     *          The data-row being predicted against.
     * @param prediction
     *          The learner's prediction for the row.
     */
    fun postPredict(learner : L, learnerIndex: Int, row: DataRow, prediction: DataRow)

}

/**
 * Interface for classes which can calculate a metric over the course of an
 * evaluation on a batch-learner.
 *
 * @param L
 *          The type of learner that the evaluator needs to measure.
 * @param M
 *          The type of metric that the evaluator measures.
 */
interface BatchEvaluator<in L: Learner<DataBatch<*>>, out M> : Evaluator<DataBatch<*>, L, M>

/**
 * Interface for classes which can calculate a metric over the course of an
 * evaluation on a stream-learner.
 *
 * @param L
 *          The type of learner that the evaluator needs to measure.
 * @param M
 *          The type of metric that the evaluator measures.
 */
interface StreamEvaluator<in L: Learner<DataStream<*>>, out M> : Evaluator<DataStream<*>, L, M> {

    /**
     * Allows a stream evaluator to optionally attach a callback to the
     * data-stream which is called before each row is sourced by the learner.
     * This method is called before [Evaluator.preTrain].
     *
     * @param learner
     *          The learner being trained.
     * @param learnerIndex
     *          The index of the learner in the evaluation.
     * @param dataset
     *          The dataset being used to train the learner.
     * @return
     *          The callback to call before each row is iterated, if any.
     */
    fun perRowCallback(learner : L, learnerIndex: Int, dataset: DataStream<*>): EvaluationHarnessCallback?

}
