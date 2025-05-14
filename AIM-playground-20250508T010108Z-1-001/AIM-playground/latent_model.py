import tensorflow as tf

class SLACEncoder(tf.keras.Model):
    """
    Convolutional encoder network for SLAC.

    Args:
        z_dim (int): Dimension of the latent vector.
        input_channels (int): Number of input channels (e.g., 9 for 3 stacked RGB frames).
    """

    def __init__(self, z_dim, input_channels=9):
        super(SLACEncoder, self).__init__()
        self.z_dim = z_dim

        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu')

        self.flatten = tf.keras.layers.Flatten()
        self.fc_mean = tf.keras.layers.Dense(z_dim)
        self.fc_logstd = tf.keras.layers.Dense(z_dim)

    def call(self, x, training=False):
        """
        Forward pass through the encoder.
        Args:
            x (Tensor): Input image stack of shape (B, 64, 64, input_channels)

        Returns:
            z (Tensor): Sampled latent vector
            mean (Tensor): Latent distribution mean
            log_std (Tensor): Latent distribution log standard deviation
        """
        x = tf.cast(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        std = tf.exp(log_std)
        eps = tf.random.normal(shape=tf.shape(std))
        z = mean + eps * std  # Reparameterization trick

        return z, mean, log_std


class PriorNetwork(tf.keras.Model):
    """
    Latent transition model for SLAC.

    Predicts the prior distribution over the next latent variable z_{t+1},
    given the previous latent z_t and action a_t.

    Args:
        z_dim (int): Dimension of the latent space.
        action_dim (int): Dimension of the action space.
    """

    def __init__(self, z_dim, action_dim):
        super(PriorNetwork, self).__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim

        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')

        self.fc_mean = tf.keras.layers.Dense(z_dim)
        self.fc_logstd = tf.keras.layers.Dense(z_dim)

    def call(self, z, a):
        """
        Args:
            z (Tensor): Previous latent state z_t, shape (B, z_dim)
            a (Tensor): Action a_t, shape (B, action_dim)

        Returns:
            z_sample: Sampled next latent vector z_{t+1}
            mean: Mean of predicted latent distribution
            log_std: Log std-dev of predicted latent distribution
        """
        x = tf.concat([z, a], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        std = tf.exp(log_std)
        eps = tf.random.normal(shape=tf.shape(std))
        z_sample = mean + std * eps  # Reparameterization trick

        return z_sample, mean, log_std



class Decoder(tf.keras.Model):
    """
    Decoder network that reconstructs images from latent vectors.

    Args:
        z_dim (int): Dimension of the latent space.
        output_shape (tuple): Target image shape, default (64, 64, 3)
    """

    def __init__(self, z_dim, output_shape=(64, 64, 3)):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.image_shape = output_shape

        self.fc = tf.keras.layers.Dense(8 * 8 * 128, activation='relu')
        self.reshape = tf.keras.layers.Reshape((8, 8, 128))

        self.deconv1 = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu')
        self.deconv3 = tf.keras.layers.Conv2DTranspose(output_shape[-1], 4, strides=2, padding='same', activation='sigmoid')

    def call(self, z):
        """
        Args:
            z (Tensor): Latent vector of shape (B, z_dim)

        Returns:
            Tensor: Reconstructed image of shape (B, 64, 64, 3)
        """
        x = self.fc(z)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class LatentModel(tf.keras.Model):
    """
    Wrapper for the encoder, prior, and decoder used in SLAC.

    Computes the ELBO loss:
        ELBO = -log p(x_{t+1} | z_{t+1}) + KL[q(z_{t+1}) || p(z_{t+1})]
    """

    def __init__(self, z_dim, action_dim, input_channels=9, image_shape=(64, 64, 3)):
        super(LatentModel, self).__init__()
        self.encoder = SLACEncoder(z_dim=z_dim, input_channels=input_channels)
        self.prior = PriorNetwork(z_dim=z_dim, action_dim=action_dim)
        self.decoder = Decoder(z_dim=z_dim, output_shape=image_shape)

    def compute_elbo(self, x_tp1, a_t, z_t):
        """
        Compute the ELBO loss using:
        - posterior q(z_tp1 | x_tp1, z_t, a_t)
        - prior     p(z_tp1 | z_t, a_t)
        - decoder   p(x_tp1 | z_tp1)

        Args:
            x_tp1: next frame (B, 64, 64, 3)
            a_t: action (B, action_dim)
            z_t: previous latent (B, z_dim)

        Returns:
            loss, dict of components
        """
        # Encode posterior from image
        z_post, mu_post, logstd_post = self.encoder(x_tp1)

        # Sample from prior
        _, mu_prior, logstd_prior = self.prior(z_t, a_t)

        # Decode
        recon_x = self.decoder(z_post)

        # Reconstruction loss: -log p(x | z) ~ BCE loss
        recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x_tp1, recon_x))

        # KL divergence between posterior and prior (analytical)
        kl = self._kl_divergence(mu_post, logstd_post, mu_prior, logstd_prior)

        total_loss = recon_loss + kl

        return total_loss, {
            "recon_loss": recon_loss,
            "kl_loss": kl,
            "reconstructed_x": recon_x,
            "z_post": z_post
        }

    @staticmethod
    def _kl_divergence(mu_q, logstd_q, mu_p, logstd_p):
        """
        KL divergence between two Gaussians:
        q ~ N(mu_q, std_q), p ~ N(mu_p, std_p)
        """
        std_q = tf.exp(logstd_q)
        std_p = tf.exp(logstd_p)

        kl = (
            logstd_p - logstd_q
            + (tf.square(std_q) + tf.square(mu_q - mu_p)) / (2.0 * tf.square(std_p))
            - 0.5
        )
        return tf.reduce_mean(tf.reduce_sum(kl, axis=1))
