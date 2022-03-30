package maia.ml.evaluation.standard.evaluator

import maia.ml.dataset.DataRow
import maia.ml.dataset.DataStream
import maia.ml.dataset.WithColumns
import maia.ml.evaluation.EvaluationHarnessCallback
import maia.ml.evaluation.Metric
import maia.ml.evaluation.StreamEvaluator
import maia.ml.learner.Learner
import maia.ml.learner.type.AnyLearnerType
import maia.ml.learner.type.LearnerType
import kotlin.time.Duration
import kotlin.time.ExperimentalTime
import kotlin.time.TimeMark
import kotlin.time.TimeSource

/**
 * Evaluator which measures the amount of time an evaluation spends
 * initialising, training and predicting.
 *
 * @param initialise
 *          Whether to time initialisation.
 * @param train
 *          Whether to time training.
 * @param predict
 *          Whether to time predicting.
 * @param timeSource
 *          The source of time to use.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
@ExperimentalTime
class Timing constructor(
    val initialise: Boolean = false,
    val train: Boolean = false,
    val predict: Boolean = false,
    val perLearner: Boolean = true,
    val timeSource: TimeSource = TimeSource.Monotonic
) : StreamEvaluator<Learner<DataStream<*>>, Duration> {

    override val name : String
        get() {
            val flags = ArrayList<String>(3)
            if (initialise) flags.add("I")
            if (train) flags.add("T")
            if (predict) flags.add("P")
            return "Timing (${flags.joinToString(",")})"
        }

    override val type : LearnerType
        get() = AnyLearnerType

    override val metric : Metric<Duration>
        get() = if (perLearner)
            Metric.PerLearner(*accumulators)
        else
            Metric.Aggregate(accumulators.fold(Duration.ZERO, Duration::plus))

    /** The time at which a corresponding pre* method was called. */
    private lateinit var preTimes : Array<TimeMark>

    /** Accumulators of time taken so far. */
    private lateinit var accumulators: Array<Duration>

    /**
     * If the flag is true, records the start time of an operation.
     *
     * @param flag
     *          Whether to record the operation.
     */
    private inline fun start(flag: Boolean, index: Int) {
        if (flag) preTimes[index] = timeSource.markNow()
    }

    /**
     * If the flag is true, updates the metric with the duration of an
     * operation.
     *
     * @param flag
     *          Whether to record the operation.
     */
    private inline fun end(flag : Boolean, index: Int) {
        if (flag) accumulators[index] += preTimes[index].elapsedNow()
    }

    override fun preEvaluate(
        vararg learners : Learner<DataStream<*>>
    ) {
        // Reset the calculations
        preTimes = Array(learners.size) { timeSource.markNow() }
        accumulators = Array(learners.size) { Duration.ZERO }
    }

    override fun postEvaluate(vararg learners : Learner<DataStream<*>>) {}

    override fun preInitialise(
        learner : Learner<DataStream<*>>,
        learnerIndex: Int,
        headers: WithColumns
    ) = start(initialise, learnerIndex)

    override fun postInitialise(
        learner : Learner<DataStream<*>>,
        learnerIndex: Int,
        headers: WithColumns
    ) = end(initialise, learnerIndex)

    override fun preTrain(
        learner : Learner<DataStream<*>>,
        learnerIndex: Int,
        dataset : DataStream<*>
    ) = start(train, learnerIndex)

    override fun postTrain(
        learner : Learner<DataStream<*>>,
        learnerIndex: Int,
        dataset : DataStream<*>
    ) = end(train, learnerIndex)

    override fun prePredict(
        learner : Learner<DataStream<*>>,
        learnerIndex: Int,
        row : DataRow
    ) = start(predict, learnerIndex)

    override fun postPredict(
        learner : Learner<DataStream<*>>,
        learnerIndex: Int,
        row : DataRow,
        prediction : DataRow
    ) = end(predict, learnerIndex)

    override fun perRowCallback(
        learner : Learner<DataStream<*>>,
        learnerIndex : Int,
        dataset : DataStream<*>
    ) : EvaluationHarnessCallback? = null
}
