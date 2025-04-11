package com.project.training;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

/**
 * Implémentation de SoundTrainer pour les spectrogrammes
 */
public class SpectrogramSoundTrainer extends SoundTrainer {

    private int height;
    private int width;
    private int channels;
    private int outputSize;
    private double learningRate;

    /**
     * Constructeur
     * 
     * @param height Hauteur du spectrogramme
     * @param width Largeur du spectrogramme
     * @param channels Nombre de canaux (1 pour grayscale)
     * @param outputSize Taille de sortie (nombre de classes)
     * @param learningRate Taux d'apprentissage
     */
    public SpectrogramSoundTrainer(int height, int width, int channels, int outputSize, double learningRate) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.trainerType = SoundTrainerType.SPECTROGRAM;
        initializeModel();
    }

    /**
     * Constructeur par défaut
     */
    public SpectrogramSoundTrainer() {
        this(128, 128, 1, 10, 0.001); // Valeurs par défaut: 128x128 grayscale, 10 classes, learning rate 0.001
    }

    @Override
    public void initializeModel() {
        conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputSize)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
    }

    @Override
    public void train(DataSet dataSet, int epochs) {
        if (model == null) {
            initializeModel();
        }

        for (int i = 0; i < epochs; i++) {
            model.fit(dataSet);
        }
    }

    @Override
    public double evaluate(DataSet testData) {
        if (model == null) {
            throw new IllegalStateException("Le modèle n'est pas initialisé. Appelez initializeModel() d'abord.");
        }
        return model.score(testData);
    }

    @Override
    public void saveModel(String filePath) {
        if (model == null) {
            throw new IllegalStateException("Le modèle n'est pas initialisé. Appelez initializeModel() d'abord.");
        }
        
        try {
            model.save(new File(filePath), true);
        } catch (IOException e) {
            throw new RuntimeException("Erreur lors de la sauvegarde du modèle : " + e.getMessage(), e);
        }
    }

    @Override
    public void loadModel(String filePath) {
        try {
            model = MultiLayerNetwork.load(new File(filePath), true);
            conf = model.getLayerWiseConfigurations();
        } catch (IOException e) {
            throw new RuntimeException("Erreur lors du chargement du modèle : " + e.getMessage(), e);
        }
    }
}
