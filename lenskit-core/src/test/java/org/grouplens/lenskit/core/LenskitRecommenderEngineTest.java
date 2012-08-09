/*
 * LensKit, an open source recommender systems toolkit.
 * Copyright 2010-2012 Regents of the University of Minnesota and contributors
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
package org.grouplens.lenskit.core;

import com.google.common.collect.Sets;
import org.grouplens.grapht.graph.Node;
import org.grouplens.grapht.spi.CachedSatisfaction;
import org.grouplens.grapht.spi.reflect.InstanceSatisfaction;
import org.grouplens.lenskit.ItemRecommender;
import org.grouplens.lenskit.RatingPredictor;
import org.grouplens.lenskit.baseline.BaselinePredictor;
import org.grouplens.lenskit.baseline.BaselineRatingPredictor;
import org.grouplens.lenskit.baseline.ConstantPredictor;
import org.grouplens.lenskit.baseline.GlobalMeanPredictor;
import org.grouplens.lenskit.data.Event;
import org.grouplens.lenskit.data.dao.DAOFactory;
import org.grouplens.lenskit.data.dao.EventCollectionDAO;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;
import java.util.Set;

import static org.hamcrest.Matchers.*;
import static org.junit.Assert.assertThat;

/**
 * @author Michael Ekstrand
 */
public class LenskitRecommenderEngineTest {
    private LenskitRecommenderEngineFactory factory;

    @Before
    public void setup() {
        DAOFactory dao = new EventCollectionDAO.Factory(Collections.<Event>emptyList());
        factory = new LenskitRecommenderEngineFactory(dao);
    }

    @Test
    public void testBasicRec() {
        factory.bind(RatingPredictor.class)
               .to(BaselineRatingPredictor.class);
        factory.bind(ItemRecommender.class)
               .to(ScoreBasedItemRecommender.class);
        factory.bind(BaselinePredictor.class)
               .to(ConstantPredictor.class);

        LenskitRecommenderEngine engine = factory.create();
        LenskitRecommender rec = engine.open();
        try {
            assertThat(rec.getItemRecommender(),
                       instanceOf(ScoreBasedItemRecommender.class));
            assertThat(rec.getItemScorer(),
                       instanceOf(BaselineRatingPredictor.class));
            assertThat(rec.getRatingPredictor(),
                       instanceOf(BaselineRatingPredictor.class));
            assertThat(rec.get(BaselinePredictor.class),
                       instanceOf(ConstantPredictor.class));
        } finally {
            rec.close();
        }
    }

    @Test
    public void testArbitraryRoot() {
        factory.bind(BaselinePredictor.class)
               .to(ConstantPredictor.class);
        factory.addRoot(BaselinePredictor.class);

        LenskitRecommenderEngine engine = factory.create();
        LenskitRecommender rec = engine.open();
        try {
            assertThat(rec.get(BaselinePredictor.class),
                       instanceOf(ConstantPredictor.class));
        } finally {
            rec.close();
        }
    }

    @Test
    public void testSeparatePredictor() {
        factory.bind(BaselinePredictor.class)
               .to(GlobalMeanPredictor.class);
        factory.bind(RatingPredictor.class)
               .to(BaselineRatingPredictor.class);

        LenskitRecommenderEngine engine = factory.create();

        LenskitRecommender rec1 = engine.open();
        LenskitRecommender rec2 = engine.open();
        try {
            assertThat(rec1.getRatingPredictor(),
                       instanceOf(BaselineRatingPredictor.class));
            assertThat(rec2.getRatingPredictor(),
                       instanceOf(BaselineRatingPredictor.class));

            // verify that recommenders have different predictors
            assertThat(rec1.getRatingPredictor(),
                       not(sameInstance(rec2.getRatingPredictor())));

            // verify that recommenders have same baseline
            assertThat(rec1.get(BaselinePredictor.class),
                       sameInstance(rec2.get(BaselinePredictor.class)));
        } finally {
            rec1.close();
            rec2.close();
        }
    }
}