package com.project.training;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Classe abstraite pour les entraîneurs de modèles audio
 */
public abstract class SoundTrainer {

    /**
     * Enumération des types d'entraîneurs audio
     */
    public enum SoundTrainerType {
        MFCC,
        SPECTROGRAM
    }

    protected MultiLayerConfiguration conf;
    protected MultiLayerNetwork model;
    protected SoundTrainerType trainerType;

    /**
     * Initialise le modèle avec la configuration par défaut
     */
    public abstract void initializeModel();

    /**
     * Entraîne le modèle avec le dataset fourni
     * 
     * @param dataSet Données d'entraînement
     * @param epochs Nombre d'epochs
     */
    public abstract void train(DataSet dataSet, int epochs);

    /**
     * Évalue le modèle avec les données de test
     * 
     * @param testData Données de test
     * @return Score d'évaluation
     */
    public abstract double evaluate(DataSet testData);

    /**
     * Sauvegarde le modèle dans un fichier
     * 
     * @param filePath Chemin du fichier
     */
    public abstract void saveModel(String filePath);

    /**
     * Charge un modèle depuis un fichier
     * 
     * @param filePath Chemin du fichier
     */
    public abstract void loadModel(String filePath);

    /**
     * Obtient le type d'entraîneur
     * 
     * @return Type d'entraîneur
     */
    public SoundTrainerType getTrainerType() {
        return trainerType;
    }

    /**
     * Définit le type d'entraîneur
     * 
     * @param trainerType Type d'entraîneur
     */
    public void setTrainerType(SoundTrainerType trainerType) {
        this.trainerType = trainerType;
    }

    /**
     * Obtient le modèle
     * 
     * @return Modèle
     */
    public MultiLayerNetwork getModel() {
        return model;
    }

    /**
     * Définit le modèle
     * 
     * @param model Modèle
     */
    public void setModel(MultiLayerNetwork model) {
        this.model = model;
    }

    /**
     * Obtient la configuration
     * 
     * @return Configuration
     */
    public MultiLayerConfiguration getConf() {
        return conf;
    }

    /**
     * Définit la configuration
     * 
     * @param conf Configuration
     */
    public void setConf(MultiLayerConfiguration conf) {
        this.conf = conf;
    }
}
