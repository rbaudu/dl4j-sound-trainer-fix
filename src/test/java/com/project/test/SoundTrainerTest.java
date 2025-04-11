package com.project.test;

import com.project.training.MFCCSoundTrainer;
import com.project.training.SoundTrainer;
import com.project.training.SpectrogramSoundTrainer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Tests unitaires pour les classes SoundTrainer, MFCCSoundTrainer et SpectrogramSoundTrainer
 */
public class SoundTrainerTest {

    private final int MFCC_INPUT_SIZE = 13;
    private final int MFCC_OUTPUT_SIZE = 10;
    
    private final int SPECTROGRAM_HEIGHT = 128;
    private final int SPECTROGRAM_WIDTH = 128;
    private final int SPECTROGRAM_CHANNELS = 1;
    private final int SPECTROGRAM_OUTPUT_SIZE = 10;
    
    private MFCCSoundTrainer mfccTrainer;
    private SpectrogramSoundTrainer spectrogramTrainer;
    
    @Before
    public void setUp() {
        // Initialisation des trainers
        mfccTrainer = new MFCCSoundTrainer(MFCC_INPUT_SIZE, MFCC_OUTPUT_SIZE, 0.001);
        spectrogramTrainer = new SpectrogramSoundTrainer(SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH, 
                                                      SPECTROGRAM_CHANNELS, SPECTROGRAM_OUTPUT_SIZE, 0.001);
    }
    
    @Test
    public void testSoundTrainerTypeEnum() {
        Assert.assertEquals(2, SoundTrainer.SoundTrainerType.values().length);
        Assert.assertEquals(SoundTrainer.SoundTrainerType.MFCC, SoundTrainer.SoundTrainerType.valueOf("MFCC"));
        Assert.assertEquals(SoundTrainer.SoundTrainerType.SPECTROGRAM, SoundTrainer.SoundTrainerType.valueOf("SPECTROGRAM"));
    }
    
    @Test
    public void testTrainerTypeGetterSetter() {
        // Utilisation d'une classe concrète pour tester les méthodes de SoundTrainer
        SoundTrainer trainer = new MFCCSoundTrainer();
        
        Assert.assertEquals(SoundTrainer.SoundTrainerType.MFCC, trainer.getTrainerType());
        
        // Test du setter
        trainer.setTrainerType(SoundTrainer.SoundTrainerType.SPECTROGRAM);
        Assert.assertEquals(SoundTrainer.SoundTrainerType.SPECTROGRAM, trainer.getTrainerType());
    }
    
    @Test
    public void testMFCCSoundTrainerInitialize() {
        // Test de l'initialisation du modèle
        mfccTrainer.initializeModel();
        
        // Vérification que le modèle est initialisé
        Assert.assertNotNull(mfccTrainer.getModel());
        Assert.assertNotNull(mfccTrainer.getConf());
    }
    
    @Test
    public void testSpectrogramSoundTrainerInitialize() {
        // Test de l'initialisation du modèle
        spectrogramTrainer.initializeModel();
        
        // Vérification que le modèle est initialisé
        Assert.assertNotNull(spectrogramTrainer.getModel());
        Assert.assertNotNull(spectrogramTrainer.getConf());
    }
    
    @Test
    public void testMFCCSoundTrainerTraining() {
        // Création d'un DataSet fictif pour l'entraînement
        INDArray features = Nd4j.rand(new int[]{10, MFCC_INPUT_SIZE});
        INDArray labels = Nd4j.zeros(10, MFCC_OUTPUT_SIZE);
        // Assignation de la classe 0 à chaque échantillon pour simplifier
        for (int i = 0; i < 10; i++) {
            labels.putScalar(new int[]{i, 0}, 1.0);
        }
        DataSet dataSet = new DataSet(features, labels);
        
        // Entraînement du modèle
        mfccTrainer.train(dataSet, 5);
        
        // Test que le modèle peut être évalué après l'entraînement
        double score = mfccTrainer.evaluate(dataSet);
        Assert.assertTrue("Le score devrait être un nombre", !Double.isNaN(score));
    }
    
    @Test
    public void testSpectrogramSoundTrainerTraining() {
        // Création d'un DataSet fictif pour l'entraînement
        // Création d'images de spectrogramme 128x128x1
        INDArray features = Nd4j.rand(new int[]{10, SPECTROGRAM_CHANNELS, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH});
        INDArray labels = Nd4j.zeros(10, SPECTROGRAM_OUTPUT_SIZE);
        // Assignation de la classe 0 à chaque échantillon pour simplifier
        for (int i = 0; i < 10; i++) {
            labels.putScalar(new int[]{i, 0}, 1.0);
        }
        DataSet dataSet = new DataSet(features, labels);
        
        // Entraînement du modèle
        spectrogramTrainer.train(dataSet, 1); // Une seule époque pour accélérer le test
        
        // Test que le modèle peut être évalué après l'entraînement
        double score = spectrogramTrainer.evaluate(dataSet);
        Assert.assertTrue("Le score devrait être un nombre", !Double.isNaN(score));
    }
    
    @Test
    public void testMFCCSoundTrainerSaveLoad() throws IOException {
        // Création d'un répertoire temporaire pour sauvegarder le modèle
        Path tempDir = Files.createTempDirectory("model-test");
        String modelPath = tempDir.resolve("mfcc-model.zip").toString();
        
        // Initialisation et sauvegarde du modèle
        mfccTrainer.initializeModel();
        mfccTrainer.saveModel(modelPath);
        
        // Vérification que le fichier existe
        File modelFile = new File(modelPath);
        Assert.assertTrue("Le fichier du modèle devrait exister", modelFile.exists());
        
        // Création d'un nouveau trainer et chargement du modèle
        MFCCSoundTrainer newTrainer = new MFCCSoundTrainer();
        newTrainer.loadModel(modelPath);
        
        // Vérification que le modèle est chargé
        Assert.assertNotNull(newTrainer.getModel());
        Assert.assertNotNull(newTrainer.getConf());
        
        // Nettoyage
        modelFile.delete();
        tempDir.toFile().delete();
    }
}
