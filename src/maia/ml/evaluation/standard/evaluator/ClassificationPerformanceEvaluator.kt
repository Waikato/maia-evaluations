package maia.ml.evaluation.standard.evaluator

import maia.ml.dataset.DataRow
import maia.ml.dataset.DataStream
import maia.ml.dataset.WithColumns
import maia.ml.dataset.error.MissingValue
import maia.ml.dataset.type.standard.Nominal
import maia.ml.dataset.util.weight
import maia.ml.evaluation.EvaluationHarnessCallback
import maia.ml.evaluation.Metric
import maia.ml.evaluation.StreamEvaluator
import maia.ml.learner.Learner
import maia.ml.learner.type.Classifier
import maia.ml.learner.type.LearnerType
import maia.ml.learner.type.SingleTarget
import maia.ml.learner.type.intersectionOf
import maia.util.assertType

/**
 * Evaluator which measures multiple forms of accuracy over the course
 * of an evaluation on a single-target classifier. Based on MOA's
 * moa.evaluation.BasicClassificationPerformanceEvaluator.
 *
 * @param precisionRecall
 *          Whether to output the overall precision/recall/F1 statistics.
 * @param precisionPerClass
 *          Whether to output the precision per-class.
 * @param recallPerClass
 *          Whether to output the recall per-class.
 * @param f1PerClass
 *          Whether to output the F1 statistic per-class.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
open class ClassificationPerformanceEvaluator(
    val precisionRecall: Boolean = false,
    val precisionPerClass: Boolean = false,
    val recallPerClass: Boolean = false,
    val f1PerClass: Boolean = false
): StreamEvaluator<Learner<DataStream<*>>, Double> {

    interface Estimator {
        fun add(value: Double)
        val value: Double
    }

    class BasicEstimator : Estimator {
        private var len: Double = 0.0
        private var sum: Double = 0.0

        override fun add(value : Double) {
            if (value.isNaN()) return
            this.sum += value
            this.len++
        }

        override val value : Double
            get() = sum / len
    }

    protected fun newEstimator(): Estimator {
        return BasicEstimator()
    }

    inner class Stats(
        learner: Learner<DataStream<*>>
    ) {
        val predictRepr = assertType<Nominal<*, *, *, *>>(learner.predictOutputHeaders[0].type).indexRepresentation
        val trainRepr = assertType<Nominal<*, *, *, *>>(learner.trainHeaders[predictRepr.columnName]!!.type).indexRepresentation
        val numClasses = predictRepr.dataType.numCategories
        val rowKappa : Array<Estimator> = Array(numClasses) { newEstimator() }
        val columnKappa : Array<Estimator> = Array(numClasses) { newEstimator() }
        val precision : Array<Estimator> = Array(numClasses) { newEstimator() }
        val recall : Array<Estimator> = Array(numClasses) { newEstimator() }
        val weightCorrect : Estimator = newEstimator()
        val weightCorrectNoChangeClassifier : Estimator = newEstimator()
        val weightMajorityClassifier : Estimator = newEstimator()
        var lastSeenClass = 0
        var totalWeightObserved = 0.0

        val majorityClass: Int
            get() {
                return (0 until numClasses).maxByOrNull {
                    columnKappa[it].value
                }!!
            }

        val fractionCorrectlyClassified: Double
            get() = weightCorrect.value

        val kappaStatistic: Double
            get() {
                if (totalWeightObserved == 0.0) return 0.0
                val pc = (0 until numClasses).fold(0.0) { acc, categoryIndex ->
                    acc + rowKappa[categoryIndex].value * columnKappa[categoryIndex].value
                }
                return (fractionCorrectlyClassified - pc) / (1.0 - pc)
            }

        val kappaTemporalStatistic: Double
            get() {
                if (totalWeightObserved == 0.0) return 0.0
                val pc = weightCorrectNoChangeClassifier.value
                return (fractionCorrectlyClassified - pc) / (1.0 - pc)
            }

        val kappaMStatistic: Double
            get() {
                if (totalWeightObserved == 0.0) return 0.0
                val pc = weightMajorityClassifier.value
                return (fractionCorrectlyClassified - pc) / (1.0 - pc)
            }

        val precisionStatistic: Double
            get() = precision.fold(0.0) { acc, ck -> acc + ck.value } / precision.size

        fun precisionStatistic(numClass: Int): Double {
            return precision[numClass].value
        }

        val recallStatistic: Double
            get() = recall.fold(0.0) { acc, ck -> acc + ck.value } / recall.size

        fun recallStatistic(numClass: Int): Double {
            return recall[numClass].value
        }

        private fun calcF1(precision: Double, recall: Double): Double {
            return 2 * (precision * recall) / (precision + recall)
        }

        val f1Statistic: Double
            get() = calcF1(precisionStatistic, recallStatistic)

        fun f1Statistic(numClass: Int) = calcF1(
            precisionStatistic(numClass),
            recallStatistic(numClass)
        )
    }

    private lateinit var learnerStats: Array<Stats?>

    override val name : String = "Classification Performance"

    final override val type : LearnerType = intersectionOf(SingleTarget, Classifier)

    override val metric : Metric<Double>
        get() {
            return Metric.Group(
                LinkedHashMap<String, Metric<Double>>().apply {
                    this["Classified Instances"] = perLearnerMetric { totalWeightObserved }
                    this["Classifications Correct (percent)"] = perLearnerMetric { fractionCorrectlyClassified * 100 }
                    this["Kappa Statistic (percent)"] = perLearnerMetric { kappaStatistic * 100 }
                    this["Kappa Temporal Statistic (percent)"] = perLearnerMetric { kappaTemporalStatistic * 100 }
                    this["Kappa M Statistic (percent)"] = perLearnerMetric { kappaMStatistic * 100 }
                    if (precisionRecall)
                        this["F1 Score (percent)"] = perLearnerMetric { f1Statistic * 100 }
                    if (f1PerClass)
                        this["F1 Score Per Class (percent)"] = perLearnerGroupMetric { f1Statistic(it) * 100 }
                    if (precisionRecall)
                        this["Precision (percent)"] = perLearnerMetric { precisionStatistic * 100 }
                    if (precisionPerClass)
                        this["Precision Per Class (percent)"] = perLearnerGroupMetric { precisionStatistic(it) * 100 }
                    if (precisionRecall)
                        this["Recall (percent)"] = perLearnerMetric { recallStatistic * 100 }
                    if (recallPerClass)
                        this["Recall Per Class (percent)"] = perLearnerGroupMetric { recallStatistic(it) * 100 }
                }
            )
        }

    private fun perLearnerMetric(block: Stats.() -> Double): Metric<Double> {
        return if (learnerStats.size > 1)
            Metric.PerLearner(*learnerStats.map { it!!.block() }.toTypedArray())
        else
            Metric.Aggregate(learnerStats[0]!!.block())
    }

    private fun perLearnerGroupMetric(block: Stats.(Int) -> Double): Metric<Double> {
        return if (learnerStats.size > 1)
            Metric.PerLearnerGroup(
                *learnerStats.map {
                    LinkedHashMap<String, Double>().apply {
                        val dataType = it!!.predictRepr.dataType
                        for (index in dataType.categoryIndices) {
                            this["Class ${dataType[index]}"] = it.block(index)
                        }
                    }
                }.toTypedArray()
            )
        else
            Metric.Group(
                LinkedHashMap<String, Metric<Double>>().apply {
                    val stats = learnerStats[0]!!
                    val dataType = stats.predictRepr.dataType
                    for (index in dataType.categoryIndices) {
                        this["Class ${dataType[index]}"] = Metric.Aggregate(stats.block(index))
                    }
                }
            )
    }

    override fun preEvaluate(vararg learners : Learner<DataStream<*>>) {
        learnerStats = Array(learners.size) { null }
    }

    override fun postEvaluate(vararg learners : Learner<DataStream<*>>) {
    }

    override fun preInitialise(
        learner : Learner<DataStream<*>>,
        learnerIndex : Int,
        headers : WithColumns
    ) {
    }

    override fun postInitialise(
        learner : Learner<DataStream<*>>,
        learnerIndex : Int,
        headers : WithColumns
    ) {
        learnerStats[learnerIndex] = Stats(learner)
    }

    override fun preTrain(
        learner : Learner<DataStream<*>>,
        learnerIndex : Int,
        dataset : DataStream<*>
    ) {
    }

    override fun postTrain(
        learner : Learner<DataStream<*>>,
        learnerIndex : Int,
        dataset : DataStream<*>
    ) {
    }

    override fun prePredict(
        learner : Learner<DataStream<*>>,
        learnerIndex : Int,
        row : DataRow
    ) {
    }

    override fun postPredict(
        learner : Learner<DataStream<*>>,
        learnerIndex : Int,
        row : DataRow,
        prediction : DataRow
    ) {
        val weight = row.weight

        if (weight == 0.0) return

        val stats = learnerStats[learnerIndex]!!

        val trueClass = try {
            row.getValue(stats.trainRepr)
        } catch (e: MissingValue) {
            return
        }

        val predictedClass = prediction.getValue(stats.predictRepr)

        stats.totalWeightObserved += weight
        stats.weightCorrect.add(if (predictedClass == trueClass) weight else 0.0)

        for (categoryIndex in 0 until stats.numClasses) {
            stats.rowKappa[categoryIndex].add(if (predictedClass == categoryIndex) weight else 0.0)
            stats.columnKappa[categoryIndex].add(if (trueClass == categoryIndex) weight else 0.0)
            stats.precision[categoryIndex].add(
                if (predictedClass != categoryIndex)
                    Double.NaN
                else if (predictedClass == trueClass)
                    weight
                else
                    0.0
            )
            stats.recall[categoryIndex].add(
                if (trueClass != categoryIndex)
                    Double.NaN
                else if (predictedClass == trueClass)
                    weight
                else
                    0.0
            )
        }

        stats.weightCorrectNoChangeClassifier.add(if (stats.lastSeenClass == trueClass) weight else 0.0)
        stats.weightMajorityClassifier.add(if (stats.majorityClass == trueClass) weight else 0.0)
        stats.lastSeenClass = trueClass
    }

    override fun perRowCallback(
        learner : Learner<DataStream<*>>,
        learnerIndex : Int,
        dataset : DataStream<*>
    ) : EvaluationHarnessCallback? = null


}
