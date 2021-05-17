#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
from unittest import mock  #type: ignore

from meld.remd import adaptor


class TestAcceptanceCounter(unittest.TestCase):
    def setUp(self):
        self.counter = adaptor._AcceptanceCounter(3)
    
    def test_should_give_zero_with_no_success(self):
        result = self.counter.get_acceptance_probabilities()
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.0)
    
    def test_should_give_100_percent_with_all_success(self):
        self.counter.update(0, True)
        result = self.counter.get_acceptance_probabilities()
        self.assertAlmostEqual(result[0], 1.0)

    def test_should_give_50_percent_with_half_success(self):
        self.counter.update(0, True)
        self.counter.update(0, False)
        result = self.counter.get_acceptance_probabilities()
        self.assertAlmostEqual(result[0], 0.5)


class TestAdaptationUsesPolicy(unittest.TestCase):
    "tests to see if adaptor delegates decisions to adapt or reset to the adapation_policy"

    def setUp(self):
        self.mock_adapt_policy = mock.Mock(spec_set=adaptor.AdaptationPolicy)
        self.adaptor = adaptor.EqualAcceptanceAdaptor(
            n_replicas=3, adaptation_policy=self.mock_adapt_policy
        )

    def test_calls_should_adapt(self):
        "the adaptor should ask the adaptation_policy if it should adapt this step"
        STEP = 10

        self.adaptor.adapt([0., 0.5, 1.], STEP)

        self.mock_adapt_policy.should_adapt.assert_called_once_with(STEP)

    def test_calls_reset(self):
        "if the adaptation_policy says so, we should reset"
        STEP = 10
        # False, True => No adapt, but reset
        self.mock_adapt_policy.should_adapt.return_value = adaptor.AdaptationRequired(
            False, True
        )
        # set one success; this should be removed by the reset
        self.adaptor.update(0, True)

        self.adaptor.adapt([0., 0.5, 1.], STEP)

        self.assertEqual(self.adaptor.successes[0], 0)

    def test_no_adapt_unless_policy_says(self):
        "if the adaptation_policy says not to adapt, we should not adapt"
        STEP = 10
        # setup some fake updates
        for i in range(10):
            self.adaptor.update(0, True)
            self.adaptor.update(1, False)
        for i in range(5):
            self.adaptor.update(0, False)
            self.adaptor.update(1, True)
        # adpatation_policy says not to adapt
        self.mock_adapt_policy.should_adapt.return_value = adaptor.AdaptationRequired(
            False, False
        )

        results = self.adaptor.adapt([0.0, 0.5, 1.0], step=STEP)

        # we shouldn't adapt
        self.assertEqual(results[0], 0.0)
        self.assertEqual(results[1], 0.5)
        self.assertEqual(results[2], 1.0)

    def test_adapt_if_policy_says(self):
        "if the adaptation_policy says to adapt, we should adapt"
        STEP = 10
        # setup some fake updates
        for i in range(10):
            self.adaptor.update(0, True)
            self.adaptor.update(1, False)
        for i in range(5):
            self.adaptor.update(0, False)
            self.adaptor.update(1, True)
        # adpatation_policy says to adapt
        self.mock_adapt_policy.should_adapt.return_value = adaptor.AdaptationRequired(
            True, False
        )

        results = self.adaptor.adapt([0.0, 0.5, 1.0], step=STEP)

        # we should adapt
        self.assertEqual(results[0], 0.0)
        self.assertAlmostEqual(results[1], 0.598, places=3)
        self.assertEqual(results[2], 1.0)


class TestTwoReplicas(unittest.TestCase):
    def setUp(self):
        self.mock_adapt_policy = mock.Mock(spec_set=adaptor.AdaptationPolicy)
        self.mock_adapt_policy.should_adapt.return_value = adaptor.AdaptationRequired(
            True, False
        )
        self.adaptor = adaptor.EqualAcceptanceAdaptor(
            n_replicas=2, adaptation_policy=self.mock_adapt_policy
        )

    def test_update_should_fail_with_bad_i(self):
        "update should throw an assertion error if replica index i is out of range"
        with self.assertRaises(AssertionError):
            # 3 is out of range for two replicas
            self.adaptor.update(3, True)

    def test_endpoints_default(self):
        "the two values of lambda will always be 0 and 1 by default"
        results = self.adaptor.adapt([0., 1.], step=0)
        self.assertEqual(results, [0., 1.])

    def test_endpoints_always(self):
        "the two values of lambda will always be 0 and 1"
        # setup some fake updates
        for i in range(10):
            self.adaptor.update(0, True)
        for i in range(5):
            self.adaptor.update(0, False)

        results = self.adaptor.adapt([0., 1.], step=0)

        self.assertEqual(results, [0., 1.])


class TestThreeReplicas(unittest.TestCase):
    def setUp(self):
        self.mock_adapt_policy = mock.Mock(spec_set=adaptor.AdaptationPolicy)
        self.mock_adapt_policy.should_adapt.return_value = adaptor.AdaptationRequired(
            True, False
        )
        self.adaptor = adaptor.EqualAcceptanceAdaptor(
            n_replicas=3, adaptation_policy=self.mock_adapt_policy
        )

    def test_should_be_even_with_no_data(self):
        "should give even spacing by default"
        results = self.adaptor.adapt([0.0, 0.5, 1.0], step=0)

        self.assertEqual(results, [0.0, 0.5, 1.0])

    def test_after_updates(self):
        "the values of lambda should be correct after updates"
        # setup some fake updates
        for i in range(10):
            self.adaptor.update(0, True)
            self.adaptor.update(1, False)
        for i in range(5):
            self.adaptor.update(0, False)
            self.adaptor.update(1, True)

        results = self.adaptor.adapt([0.0, 0.5, 1.0], step=0)

        # ln Acc = -L^2 / 2
        # sqrt(-2 * ln Acc) = L
        # acc_0_1 = 0.666 -> L = 0.9
        # acc_1_2 = 0.333 -> L = 1.48
        # L_tot = 2.38 -> L_0_1 = L_1_2 = 1.19
        # lambda_0 = 0
        # lambda_1 = 0.598
        # lambda_2 = 1
        self.assertEqual(results[0], 0.)
        self.assertAlmostEqual(results[1], 0.598, places=3)
        self.assertEqual(results[2], 1.)

    def test_reset(self):
        "calling reset should make the adaptor forget about past results"
        # setup some fake updates
        for i in range(10):
            self.adaptor.update(0, True)
            self.adaptor.update(1, False)
        for i in range(5):
            self.adaptor.update(0, False)
            self.adaptor.update(1, True)
        # now reset the adaptor
        self.adaptor.reset()

        results = self.adaptor.adapt([0.0, 0.5, 1.0], step=0)

        # because we reset, we whould forget about the updates
        self.assertEqual(results, [0.0, 0.5, 1.0])


class TestMinimum(unittest.TestCase):
    def test_minimum(self):
        "the minimum value should work correctly"
        mock_adapt_policy = mock.Mock(spec_set=adaptor.AdaptationPolicy)
        mock_adapt_policy.should_adapt.return_value = adaptor.AdaptationRequired(
            True, False
        )
        a = adaptor.EqualAcceptanceAdaptor(
            n_replicas=3, adaptation_policy=mock_adapt_policy, min_acc_prob=0.5
        )
        # acc_0_1 = 0.3
        # acc_1_2 = 0.5
        a.update(0, False)
        a.update(0, False)
        a.update(0, True)
        a.update(1, False)
        a.update(1, True)

        results = a.adapt([0., 0.5, 1.], step=0)

        # min_acc_prob should raise acc_0_1 to 0.5
        # which will give no adaptation
        self.assertEqual(results, [0., 0.5, 1.])


class TestAdaptationPolicyNoGrowth(unittest.TestCase):
    def setUp(self):
        self.BURN_IN = 50
        self.ADAPT_EVERY = 100
        self.policy = adaptor.AdaptationPolicy(1.0, self.BURN_IN, self.ADAPT_EVERY)
        self.mock_adaptor = mock.Mock(adaptor.EqualAcceptanceAdaptor)

    def test_nothing_happens_first_49(self):
        for i in range(self.BURN_IN):
            results = self.policy.should_adapt(i)
            self.assertEqual(results.adapt_now, False)
            self.assertEqual(results.reset_now, False)

    def test_should_reset_step_50(self):
        "should want to reset at step 50"
        results = self.policy.should_adapt(self.BURN_IN)

        self.assertEqual(results.adapt_now, False)
        self.assertEqual(results.reset_now, True)

    def test_nothing_between_51_and_149(self):
        "should not want to do anything from 51 to 149"
        # this will reset
        self.policy.should_adapt(self.BURN_IN)

        # these should do nothing
        for i in range(self.BURN_IN + 1, self.BURN_IN + self.ADAPT_EVERY):
            results = self.policy.should_adapt(i)
            self.assertEqual(results.adapt_now, False)
            self.assertEqual(results.reset_now, False)

    def test_should_adapt_and_reset_at_150(self):
        "should want to reset and adapt at step 150"
        # this will reset
        self.policy.should_adapt(self.BURN_IN)

        # this should adapt and reset
        results = self.policy.should_adapt(self.BURN_IN + self.ADAPT_EVERY)

        self.assertEqual(results.adapt_now, True)
        self.assertEqual(results.reset_now, True)

    def test_nothing_between_151_and_249(self):
        "should not want to do anything from 151 to 249"
        self.policy.should_adapt(self.BURN_IN)
        self.policy.should_adapt(self.BURN_IN + self.ADAPT_EVERY)

        # these should do nothing
        start = self.BURN_IN + self.ADAPT_EVERY + 1
        end = start + self.ADAPT_EVERY - 1
        for i in range(start, end):
            results = self.policy.should_adapt(i)
            self.assertEqual(results.adapt_now, False)
            self.assertEqual(results.reset_now, False)

    def test_should_adapt_and_reset_at_250(self):
        "should want to reset and adapt at step 250"
        # this will reset
        self.policy.should_adapt(self.BURN_IN)
        # this will adapt and reset
        self.policy.should_adapt(self.BURN_IN + self.ADAPT_EVERY)

        # this should adapt and reset
        results = self.policy.should_adapt(self.BURN_IN + 2 * self.ADAPT_EVERY)

        self.assertEqual(results.adapt_now, True)
        self.assertEqual(results.adapt_now, True)


class TestAdaptationPolicyWithDoubling(unittest.TestCase):
    def setUp(self):
        self.ADAPT_EVERY = 100
        self.policy = adaptor.AdaptationPolicy(2.0, 0, self.ADAPT_EVERY)
        self.mock_adaptor = mock.Mock(adaptor.EqualAcceptanceAdaptor)

    def test_adapt_at_100(self):
        "should adapt at step 100"
        results = self.policy.should_adapt(100)

        self.assertEqual(results.adapt_now, True)
        self.assertEqual(results.reset_now, True)

    def test_should_not_adapt_at_200(self):
        "shoud not adapt at step 200"
        self.policy.should_adapt(100)

        results = self.policy.should_adapt(200)

        self.assertEqual(results.adapt_now, False)
        self.assertEqual(results.reset_now, False)

    def test_should_adapt_at_300(self):
        "should adapt at step 300"
        self.policy.should_adapt(100)

        results = self.policy.should_adapt(300)

        self.assertEqual(results.adapt_now, True)
        self.assertEqual(results.reset_now, True)
