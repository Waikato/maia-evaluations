package maia.ml.evaluation

import maia.ml.dataset.arff.load
import maia.ml.evaluation.standard.evaluation.EvaluatePrequential
import maia.ml.evaluation.standard.evaluator.ClassificationPerformanceEvaluator
import maia.ml.evaluation.standard.evaluator.Timing
import maia.ml.learner.standard.hoeffdingtree.HoeffdingTree
import maia.util.getResourceStatic
import kotlin.test.Test
import kotlin.time.ExperimentalTime

/**
 * Performs unit tests for evaluations.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
class EvaluateTest {

    @ExperimentalTime
    @Test
    fun testEvaluatePrequentialTiming() {

        val timingEvaluators = GroupEvaluator.create(
            "Timings",
            Timing(initialise = true, perLearner = false),
            Timing(train = true, perLearner = false),
            Timing(predict = true, perLearner = false),
            Timing(true, true, true, perLearner = false)
        )

        val performanceEvaluator = ClassificationPerformanceEvaluator(true, true, true, true)

        val datasetURL = getResourceStatic("/electricity-normalized.arff")
            ?: throw Exception("Could not find resource '/electricity-normalized.arff'")
        val dataset = load(datasetURL.file)

        val learner = HoeffdingTree()

        val metrics = arrayOf(learner).evaluate(
            EvaluatePrequential(dataset),
            GroupEvaluator.create("Evaluators", performanceEvaluator, timingEvaluators)
        )

        println("Timing")
        println("======")
        println(metrics)
    }
}
