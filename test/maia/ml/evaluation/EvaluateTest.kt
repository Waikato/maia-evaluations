package maia.ml.evaluation

import maia.ml.dataset.arff.load
import maia.ml.evaluation.standard.evaluation.EvaluatePrequential
import maia.ml.evaluation.standard.evaluator.Timing
import maia.ml.learner.standard.hoeffdingtree.HoeffdingTree
import maia.util.formatPretty
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

        val evaluators = GroupEvaluator.create(
            "Timings",
            Timing(initialise = true, perLearner = false),
            Timing(train = true, perLearner = false),
            Timing(predict = true, perLearner = false),
            Timing(true, true, true, perLearner = false)
        )

        val datasetURL = getResourceStatic("/electricity-normalized.arff")
            ?: throw Exception("Could not find resource '/electricity-normalized.arff'")
        val dataset = load(datasetURL.file)

        val learner = HoeffdingTree()

        val timing = arrayOf(learner).evaluate(
            EvaluatePrequential(dataset),
            evaluators
        )

        println("Timing")
        println("======")
        println(timing)
    }
}
