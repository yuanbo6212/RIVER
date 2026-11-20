"""Analytic geometry/causality checks; these are not tracking performance data."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from river_belief import ParticleBelief
from river_geometry import (
    CameraFrame, PinholeCalibration, backproject_optical_depth,
    calibration_from_horizontal_fov, camera_frame_from_carla_pose,
    intersect_world_plane, pixel_world_rays, project_particle_belief,
    project_world, reproject_moving_pixels,
)


def calibration():
    return PinholeCalibration(640, 360, [[320., 0., 320.], [0., 320., 180.], [0., 0., 1.]],
                              "analytic-640x360-horizontal-fov90")


def frame(center=(0., 0., 0.), basis=None, timestamp=0., handedness="right", name="analytic-world"):
    return CameraFrame(center, np.eye(3) if basis is None else basis,
                       timestamp, name, handedness)


class GeometryTests(unittest.TestCase):
    def test_known_pixel_and_half_open_image_bounds(self):
        result = project_world([[0., 0., 10.], [-10., 0., 10.], [10., 0., 10.],
                                [0., -5.625, 10.], [0., 5.625, 10.]],
                               frame(), calibration(), near_clip=.1)
        np.testing.assert_allclose(result.pixels, [[320., 180.], [0., 180.], [640., 180.],
                                                   [320., 0.], [320., 360.]])
        np.testing.assert_array_equal(result.in_image, [True, True, False, True, False])

    def test_behind_and_near_plane_never_yield_fictitious_pixels(self):
        result = project_world([[1., 2., -1.], [0., 0., 0.], [0., 0., .1], [0., 0., .11]],
                               frame(), calibration(), near_clip=.1)
        self.assertTrue(np.isnan(result.pixels[:3]).all())
        np.testing.assert_array_equal(result.in_front, [False, False, False, True])
        self.assertFalse(result.in_image[:3].any())

    def test_translated_camera_changes_pixel_coordinates_even_for_static_target(self):
        result = reproject_moving_pixels([[320., 180.]], [10.], [[0., 0., 0.]],
                                         frame(), frame(center=(2., 0., 0.), timestamp=1.),
                                         calibration(), near_clip=.1)
        np.testing.assert_allclose(result.pixels, [[256., 180.]])

    def test_rotated_camera_reprojection_is_not_old_box_velocity_reset(self):
        # Camera optical z looks world +x, optical x points world -z.
        basis = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
        destination = frame(basis=basis, timestamp=2.)
        result = reproject_moving_pixels([[480., 180.]], [10.], [[0., 0., 0.]],
                                         frame(), destination, calibration(), near_clip=.1)
        np.testing.assert_allclose(result.optical_depth, [5.])
        np.testing.assert_allclose(result.pixels, [[-320., 180.]])
        self.assertFalse(result.in_image[0])

    def test_target_motion_and_camera_motion_are_both_accounted_for(self):
        result = reproject_moving_pixels([[320., 180.]], [10.], [[2., 0., 0.]],
                                         frame(), frame(center=(1., 0., 0.), timestamp=2.),
                                         calibration(), near_clip=.1)
        np.testing.assert_allclose(result.pixels, [[416., 180.]])

    def test_pixel_depth_roundtrip_supports_skew_and_explicit_principal_point(self):
        k = PinholeCalibration(713, 421, [[400., 3., 355.], [0., 420., 208.], [0., 0., 1.]],
                               "analytic-skew")
        points = np.array([[4., 1., 10.], [-2., 4., 17.], [0., 2., 7.]])
        source = frame(center=(-2., 1., 2.))
        result = project_world(points, source, k, near_clip=.1)
        restored = backproject_optical_depth(result.pixels, result.optical_depth, source, k)
        np.testing.assert_allclose(restored, points, atol=1e-13)

    def test_euclidean_range_is_not_optical_depth(self):
        points = backproject_optical_depth([[640., 180.]], [10.], frame(), calibration())
        np.testing.assert_allclose(points, [[10., 0., 10.]])
        self.assertGreater(np.linalg.norm(points[0]), 10.)

    def test_bearing_has_no_implicit_depth_and_unit_norm(self):
        rays = pixel_world_rays([[320., 180.], [640., 180.]], frame(), calibration())
        np.testing.assert_allclose(rays, [[0., 0., 1.], [2**-.5, 0., 2**-.5]])
        np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.)

    def test_ground_plane_intersection_marks_parallel_and_backward_invalid(self):
        points, valid = intersect_world_plane([[320., 260.], [320., 180.], [320., 100.]],
                                              frame(), calibration(),
                                              plane_point=[0., 2., 0.], plane_normal=[0., 1., 0.])
        np.testing.assert_array_equal(valid, [True, False, False])
        np.testing.assert_allclose(points[0], [0., 2., 8.])
        self.assertTrue(np.isnan(points[1:]).all())

    def test_plane_normal_scale_does_not_change_intersection(self):
        outputs = [intersect_world_plane([[400., 260.]], frame(), calibration(),
                                          plane_point=[0., 2., 0.], plane_normal=[0., n, 0.])
                   for n in (1., 99., -7.)]
        for points, valid in outputs:
            self.assertTrue(valid[0])
            np.testing.assert_allclose(points, [[2., 2., 8.]])

    def test_left_handed_world_basis_is_valid_but_parity_must_be_declared(self):
        # CARLA at zero yaw/pitch/roll: optical right=world+y, down=-z, forward=+x.
        basis = [[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]]
        left = frame(basis=basis, handedness="left", name="explicit-left-world")
        result = project_world([[10., 2., 1.]], left, calibration(), near_clip=.1)
        np.testing.assert_allclose(result.pixels, [[384., 148.]])
        with self.assertRaises(ValueError):
            frame(basis=basis, handedness="right")

    def test_world_coordinate_rigid_change_preserves_image(self):
        angle = .7
        q = np.array([[np.cos(angle), -np.sin(angle), 0.],
                      [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        shift = np.array([7., -2., 1.])
        source = frame(center=(1., 2., 3.))
        points = np.array([[2., 3., 10.], [-1., 1., 8.]])
        result1 = project_world(points, source, calibration(), near_clip=.1)
        changed = frame(center=q @ source.center_world + shift,
                        basis=q @ source.world_from_optical)
        result2 = project_world(points @ q.T + shift, changed, calibration(), near_clip=.1)
        np.testing.assert_allclose(result1.pixels, result2.pixels, atol=1e-12)

    def test_particle_frustum_mass_preserves_outside_particles_and_weights(self):
        belief = ParticleBelief([[0., 0., 10., 0., 0., 0.], [1., 0., 10., 0., 0., 0.],
                                 [100., 0., 10., 0., 0., 0.], [0., 0., -1., 0., 0., 0.]],
                                [1., 2., 3., 4.], 0.)
        result = project_particle_belief(belief, frame(), calibration(), near_clip=.1)
        self.assertAlmostEqual(result.frustum_mass, .3)
        np.testing.assert_allclose(result.belief.weights, belief.weights, atol=1e-16, rtol=1e-15)
        self.assertEqual(len(result.belief.states), 4)
        np.testing.assert_allclose(result.conditional_pixel_mean, [320.+64./3, 180.])
        self.assertAlmostEqual(result.conditional_pixel_covariance[0, 0], 32.**2 * 2./9.)

    def test_particles_move_using_elapsed_time_not_number_of_frames(self):
        belief = ParticleBelief([[0., 0., 10., 1., 0., 0.]], [1.], .5)
        result = project_particle_belief(belief, frame(timestamp=2.), calibration(), near_clip=.1)
        self.assertEqual(result.belief.timestamp, 2.)
        np.testing.assert_allclose(result.projection.pixels, [[368., 180.]])

    def test_seeded_process_noise_replay_is_repeatable(self):
        belief = ParticleBelief(np.tile([0., 0., 10., 0., 0., 0.], (100, 1)), np.ones(100), 0.)
        kwargs = dict(near_clip=.1, acceleration_spectral_density=.3)
        first = project_particle_belief(belief, frame(timestamp=1.), calibration(),
                                        rng=np.random.default_rng(53), **kwargs)
        second = project_particle_belief(belief, frame(timestamp=1.), calibration(),
                                         rng=np.random.default_rng(53), **kwargs)
        np.testing.assert_array_equal(first.belief.states, second.belief.states)
        with self.assertRaises(ValueError):
            project_particle_belief(belief, frame(timestamp=1.), calibration(), **kwargs)

    def test_zero_in_image_mass_has_no_conditional_moments(self):
        belief = ParticleBelief([[0., 0., -1., 0., 0., 0.]], [1.], 0.)
        result = project_particle_belief(belief, frame(), calibration(), near_clip=.1)
        self.assertEqual(result.frustum_mass, 0.)
        self.assertIsNone(result.conditional_pixel_mean)
        self.assertIsNone(result.conditional_pixel_covariance)

    def test_backward_time_world_mismatch_and_missing_depth_rejected(self):
        with self.assertRaises(ValueError):
            reproject_moving_pixels([[320., 180.]], [10.], [[0., 0., 0.]], frame(timestamp=2.),
                                    frame(timestamp=1.), calibration(), near_clip=.1)
        with self.assertRaises(ValueError):
            reproject_moving_pixels([[320., 180.]], [10.], [[0., 0., 0.]], frame(),
                                    frame(name="other-world"), calibration(), near_clip=.1)
        for depth in ([-1.], [0.], [float("nan")], []):
            with self.assertRaises(ValueError):
                backproject_optical_depth([[320., 180.]], depth, frame(), calibration())
        belief = ParticleBelief([[0., 0., 10., 0., 0., 0.]], [1.], 2.)
        with self.assertRaises(ValueError):
            project_particle_belief(belief, frame(timestamp=1.), calibration(), near_clip=.1)

    def test_input_mutation_and_invalid_camera_calibration_rejected(self):
        center = np.zeros(3)
        camera = frame(center=center)
        center[0] = 99.
        self.assertEqual(camera.center_world[0], 0.)
        with self.assertRaises(ValueError):
            camera.center_world[0] = 1.
        for basis in (np.diag([1., 1., 2.]), np.zeros((3, 3)), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                frame(basis=basis)
        for k in ([[0., 0., 320.], [0., 320., 180.], [0., 0., 1.]],
                  [[320., 0., 320.], [1., 320., 180.], [0., 0., 1.]]):
            with self.assertRaises(ValueError):
                PinholeCalibration(640, 360, k, "invalid")

    def test_empty_observation_arrays_have_defined_shapes(self):
        points = np.empty((0, 3))
        result = project_world(points, frame(), calibration(), near_clip=.1)
        self.assertEqual(result.pixels.shape, (0, 2))
        self.assertEqual(result.in_image.shape, (0,))
        rays = pixel_world_rays(np.empty((0, 2)), frame(), calibration())
        self.assertEqual(rays.shape, (0, 3))

    def test_horizontal_fov_matches_sensor_focal_not_legacy90_default(self):
        k = calibration_from_horizontal_fov(640, 360, np.rad2deg(2*np.arctan(2/3)),
                                             calibration_id="explicit-sensor-fov")
        np.testing.assert_allclose(k.matrix, [[480., 0., 320.], [0., 480., 180.], [0., 0., 1.]])
        legacy = calibration_from_horizontal_fov(640, 360, 90., calibration_id="legacy")
        self.assertNotEqual(k.matrix[0, 0], legacy.matrix[0, 0])
        for width, height, fov in ((True, 360, 90), (640, 360.1, 90), (640, 360, 180)):
            with self.assertRaises(ValueError):
                calibration_from_horizontal_fov(width, height, fov, calibration_id="invalid")

    def test_carla_euler_constructor_matches_independent_matrix_chain(self):
        p, y, r = np.deg2rad([-17., 83., 9.])
        ry = [[np.cos(p), 0., -np.sin(p)], [0., 1., 0.], [np.sin(p), 0., np.cos(p)]]
        rz = [[np.cos(y), -np.sin(y), 0.], [np.sin(y), np.cos(y), 0.], [0., 0., 1.]]
        rx = [[1., 0., 0.], [0., np.cos(r), np.sin(r)], [0., -np.sin(r), np.cos(r)]]
        axes = [[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]]
        expected = np.array(rz) @ np.array(ry) @ np.array(rx) @ np.array(axes)
        camera = camera_frame_from_carla_pose(
            dict(x=11., y=-2., z=7., pitch=-17., yaw=83., roll=9.), timestamp=3.,
            world_frame_id="Town05-map")
        np.testing.assert_allclose(camera.world_from_optical, expected, atol=1e-15)
        np.testing.assert_array_equal(camera.center_world, [11., -2., 7.])
        self.assertEqual(camera.world_handedness, "left")
        self.assertAlmostEqual(np.linalg.det(camera.world_from_optical), -1.)

    def test_carla_zero_yaw_camera_and_gt_payload_rejection(self):
        pose = dict(x=0., y=0., z=0., pitch=0., yaw=0., roll=0.)
        camera = camera_frame_from_carla_pose(pose, timestamp=0., world_frame_id="Town05")
        np.testing.assert_allclose(pixel_world_rays([[320., 180.]], camera, calibration()), [[1., 0., 0.]])
        pose["target_world_position"] = [9., 0., 0.]
        with self.assertRaises(ValueError):
            camera_frame_from_carla_pose(pose, timestamp=0., world_frame_id="Town05")


if __name__ == "__main__":
    unittest.main()
