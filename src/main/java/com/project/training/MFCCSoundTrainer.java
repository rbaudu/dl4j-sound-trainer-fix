package com.project.training;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

/**
 * Implémentation de SoundTrainer pour les caractéristiques MFCC (Mel-Frequency Cepstral Coefficients)
 */
public class MFCCSoundTrainer extends SoundTrainer {

    private int inputSize;
    private int outputSize;
    private double learningRate;

    /**
     * Constructeur
     * 
     * @param inputSize Taille d'entrée (nombre de coefficients MFCC)
     * @param outputSize Taille de sortie (nombre de classes)
     * @param learningRate Taux d'apprentissage
     */
    public MFCCSoundTrainer(int inputSize, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.trainerType = SoundTrainerType.MFCC;
        initializeModel();
    }

    /**
     * Constructeur par défaut
     */
    public MFCCSoundTrainer() {
        this(13, 10, 0.001); // Valeurs par défaut: 13 coefficients MFCC, 10 classes, learning rate 0.001
    }

    @Override
    public void initializeModel() {
        conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputSize)
                        .activation(Activation.SOFTMAX)
                        .build())
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
