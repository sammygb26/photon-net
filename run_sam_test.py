# Setup (configure globals)
from p_pack import globals as g

g.num_steps                 = 100
g.training_rate             = 0.1
g.save_points               = [g.num_steps]
g.reupload_freq             = 4         # reupload on layer 5
g.input_positions           = [4, 5]
g.shuffle_type              = 1         # no shuffling
g.use_extra_features        = False
g.total_features            = 10        # Including reupload
g.num_features              = 5
g.num_modes_circ            = g.num_features * 2
g.depth                     = g.num_modes_circ
g.p_and_q_encoding          = True
g.p_suc_inputs              = 1.0
g.use_symmetry_parity       = False
g.pos_allowed               = [1 for _ in range(g.num_modes_circ)]
g.input_config              = g.input_config_maker('full', g.num_modes_circ, g.p_suc_inputs)
g.aim                       = 10
g.loss_function             = 0
g.loss_metric               = 0
g.coherence                 = 1.0
g.freeze_phases             = False     # Unfree circuit (training)
g.readout_type              = 1         # Trained linear readout        
g.batch_mode                = 'mini'    # Use mini-batches
g.mini_batch_size           = 512       # Mini-batch size
g.master_key                = g.jax.random.PRNGKey(0)
g.phase_key                 = g.jax.random.PRNGKey(6)
g.shuffle_key               = g.jax.random.PRNGKey(51)
g.position_sampling         = False     # Don't sample input positions
g.dataset_name              = 'mnist_pca'
g.class_labels              = [3, 5]
g.use_binary_labels         = True
g.num_classes               = len(g.class_labels)
g.use_input_superposition   = False
g.max_photons               = 3

# Run
import pathlib

from p_pack import circ  as c
from p_pack import loss  as l
from p_pack import utils as u

if __name__ == "__main__":

    # Folders used
    log_file = 'data_log'
    data_name = 'sam_test_data'
    global_name = 'sam_test_global'

    output_folder = str(pathlib.Path.home()/'work'/'sam_test')

    # Load data
    train_set, train_labels, _, _ = g.final_load_data(num_feature=g.total_features)

    # Initialise phases
    init_phases = c.initialize_phases(
        depth           = g.depth,
        width           = g.num_modes_circ,
        reupload_freq   = g.reupload_freq,
        key_init        = g.phase_key
    )

    weights_data = g.jnp.ones(shape=init_phases.shape[0:2])
    photon_loss_scale = float(1)
    readout_weights = g.jax.random.normal(g.jax.random.PRNGKey(1), (c.max_n_combos, g.num_classes))

    # Initialise carry for training
    init_loss, (_, key) = l.loss(
        init_phases,
        train_set,
        train_labels,
        weights_data,
        photon_loss_scale,
        g.input_config,
        g.pos_allowed,
        g.master_key,
        g.loss_function,
        g.aim,
        g.reupload_freq,
        g.shuffle_type,
        g.use_input_superposition,
        readout_weights,
        g.readout_type
    )

    init_carry = (
        init_phases,
        train_set,
        train_labels,
        weights_data,
        photon_loss_scale,
        0.0 * init_phases,
        0.0 * init_phases,
        0.0 * weights_data,
        0.0 * weights_data,
        0.0 * photon_loss_scale,
        0.0 * photon_loss_scale,
        key,
        init_loss,
        readout_weights,
        0.0 * readout_weights,
        0.0 * readout_weights
    )

    u.save_run(
        log_file        = log_file,
        output_folder   = output_folder, 
        data_name       = data_name,
        global_name     = global_name,
        init_carry      = init_carry
    )
    