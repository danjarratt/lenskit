/*
 * LensKit, an open source recommender systems toolkit.
 * Copyright 2010-2013 Regents of the University of Minnesota and contributors
 * Work on LensKit has been funded by the National Science Foundation under
 * grants IIS 05-34939, 08-08692, 08-12148, and 10-17697.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

import org.grouplens.lenskit.ItemScorer
import org.grouplens.lenskit.baseline.BaselineScorer
import org.grouplens.lenskit.baseline.ItemMeanRatingItemScorer
import org.grouplens.lenskit.baseline.UserMeanBaseline
import org.grouplens.lenskit.baseline.UserMeanItemScorer
import org.grouplens.lenskit.eval.data.crossfold.RandomOrder
import org.grouplens.lenskit.eval.metrics.predict.*
import org.grouplens.lenskit.eval.metrics.topn.*
import org.grouplens.lenskit.knn.item.*
import org.grouplens.lenskit.knn.user.*

def dataDir = config['lenskit.movielens.100k']

trainTest {
    dataset crossfold("ML100K") {
        source csvfile("/project/grouplens/projects/gl-location/lenskit-data/user_ratings_100K.csv") {
            delimiter ","
        }
	order RandomOrder
	partitions 5
        holdout 5
        train 'train.%d.csv'
        test 'test.%d.csv'
    }

    externalAlgorithm("LocationAwareCosineNoFallback") {
        command "python", "/project/grouplens/projects/gl-location/lenskit-data/item_mean_cosine_nocoverage.py", "{TRAIN_DATA}", "{TEST_DATA}", "{OUTPUT}"
        workDir config.scriptDir
    }

    algorithm("GlobalMean") {
        bind ItemScorer to GlobalMeanRatingItemScorer
    }
/*
    algorithm("ItemMean") {
        bind ItemScorer to ItemMeanRatingItemScorer
    }

    algorithm("PersMean") {
        bind ItemScorer to UserMeanItemScorer
	bind (UserMeanBaseline, ItemScorer) to ItemMeanRatingItemScorer
    }

    algorithm("ItemItemNormalizedWithFallback") {
        bind ItemScorer to ItemItemScorer
	bind UserVectorNormalizer to BaselineSubtractingUserVectorNormalizer
        within (UserVectorNormalizer) {
            bind (BaselineScorer, ItemScorer) to ItemMeanRatingItemScorer
        } 
    }
    algorithm("ItemItemNormalizedNoFallback") {
        bind ItemScorer to ItemItemScorer
	bind UserVectorNormalizer to BaselineSubtractingUserVectorNormalizer
	at (RatingPredictor) {
    	    bind (BaselineScorer, ItemScorer) to null
	}
    }
*/
    metric CoveragePredictMetric
    metric RMSEPredictMetric
    metric MAEPredictMetric
    metric NDCGPredictMetric
    metric HLUtilityPredictMetric
    metric topNnDCG {
        listSize 10
        candidates ItemSelectors.allItems()
        exclude ItemSelectors.trainingItems()
    }

    output 'results.csv'
    userOutput 'users.csv'
    predictOutput 'predictions.csv'
}
