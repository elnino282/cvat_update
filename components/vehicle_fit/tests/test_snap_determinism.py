"""Regression for pinned SNAP pooling ignoring the top-level shuffle flag."""
from types import SimpleNamespace
import unittest

from vehicle_fit.snap_backend import disable_serialization_shuffle


class SerializationOrderTests(unittest.TestCase):
    def test_eval_pooling_inherits_fixed_order_without_touching_weights(self):
        # Mirrors upstream: root flag false; independently constructed pool true.
        weight = object()
        outer = SimpleNamespace(shuffle_orders=False, weight=weight)
        pooling = SimpleNamespace(shuffle_orders=True, weight=weight)
        decoder = SimpleNamespace(weight=weight)
        model = SimpleNamespace(named_modules=lambda: iter([
            ('backbone', outer), ('backbone.enc.down', pooling), ('decoder', decoder),
        ]))
        self.assertTrue(pooling.shuffle_orders)  # Original bug survives eval().
        configured = disable_serialization_shuffle(model)
        self.assertEqual(configured, ['backbone', 'backbone.enc.down'])
        self.assertFalse(outer.shuffle_orders)
        self.assertFalse(pooling.shuffle_orders)
        self.assertFalse(hasattr(decoder, 'shuffle_orders'))
        self.assertIs(pooling.weight, weight)
        self.assertEqual(disable_serialization_shuffle(model), configured)

    def test_every_nested_pool_is_recorded_in_provenance(self):
        pools = [SimpleNamespace(shuffle_orders=True) for _ in range(4)]
        model = SimpleNamespace(named_modules=lambda: iter((f'enc.{i}.down', p) for i, p in enumerate(pools)))
        names = disable_serialization_shuffle(model)
        self.assertEqual(len(names), 4)
        self.assertTrue(all(not p.shuffle_orders for p in pools))


if __name__ == '__main__':
    unittest.main()
