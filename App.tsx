import { StatusBar } from 'expo-status-bar';
import { Button, InteractionManager, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { bundleResourceIO,cameraWithTensors  } from '@tensorflow/tfjs-react-native'
import * as tf from '@tensorflow/tfjs';
import { useCallback, useEffect, useRef, useState } from 'react';
import {Camera} from 'expo-camera'
import { ExpoWebGLRenderingContext } from 'expo-gl';
import { Tensor3D } from '@tensorflow/tfjs';

const TensorCamera = cameraWithTensors(Camera)

const labels = [
  'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

const loadModel = async () => {
  try {
    console.log('awaiting ready..')
    await tf.ready();
    
    console.log('loading model..')
    const modelJson = require('./assets/tensorflow/model.json');
    const modelWeights: number[] = [
      require(`./assets/tensorflow/group1-shard1of22.bin`),
      require(`./assets/tensorflow/group1-shard2of22.bin`),
      require(`./assets/tensorflow/group1-shard3of22.bin`),
      require(`./assets/tensorflow/group1-shard4of22.bin`),
      require(`./assets/tensorflow/group1-shard5of22.bin`),
      require(`./assets/tensorflow/group1-shard6of22.bin`),
      require(`./assets/tensorflow/group1-shard7of22.bin`),
      require(`./assets/tensorflow/group1-shard8of22.bin`),
      require(`./assets/tensorflow/group1-shard9of22.bin`),
      require(`./assets/tensorflow/group1-shard10of22.bin`),
      require(`./assets/tensorflow/group1-shard11of22.bin`),
      require(`./assets/tensorflow/group1-shard12of22.bin`),
      require(`./assets/tensorflow/group1-shard13of22.bin`),
      require(`./assets/tensorflow/group1-shard14of22.bin`),
      require(`./assets/tensorflow/group1-shard15of22.bin`),
      require(`./assets/tensorflow/group1-shard16of22.bin`),
      require(`./assets/tensorflow/group1-shard17of22.bin`),
      require(`./assets/tensorflow/group1-shard18of22.bin`),
      require(`./assets/tensorflow/group1-shard19of22.bin`),
      require(`./assets/tensorflow/group1-shard20of22.bin`),
      require(`./assets/tensorflow/group1-shard21of22.bin`),
      require(`./assets/tensorflow/group1-shard22of22.bin`),
    ];
    
    return tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
  } catch (error) {
    console.error(error);
    throw error
  }
}

export default function App() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [cameraReady, setCameraReady] = useState<boolean>(false);
  const {width} = useWindowDimensions();
  const [topPredictions, setTopPredictions] = useState<{label: string, probability: number}[]>([]);
  // const cameraRef = useRef<typeof TensorCamera>(null);
  
  useEffect(() => {
    loadModel().then(setModel);

    Camera.requestCameraPermissionsAsync().then(({ status }) => setHasPermission(status === 'granted'));
  }, [])

  useEffect(() => {
    if(model && cameraReady){

    }
  }, [])

  /*const takePicture = useCallback(async () => {
    if(cameraReady && model){
      const pic = await cameraRef.current?.takePictureAsync();

      if(pic){
        const image = tf.browser.fromPixels(pic.uri);
        const imageResized = tf.image.resizeBilinear(image, [28, 28]);
        const imageNormalized = imageResized.div(255.0);
        const imageTensor = imageNormalized.expandDims(0);
        const prediction = await model.predict(imageTensor).data();
        const predictionIndex = prediction.indexOf(Math.max(...prediction));
        console.log(predictionIndex)
    }
  }, [])*/
  const handleCameraStream = useCallback((images: IterableIterator<tf.Tensor3D>, updateCameraPreview: () => void, gl: ExpoWebGLRenderingContext, cameraTexture: WebGLTexture) => {
    console.log('handleCameraStream')

    let lastProcessing = 0;
    
    const loop = async () => {
      if(model && lastProcessing + 1000 < Date.now()){
        lastProcessing = Date.now();
        console.log('lastProcessing', lastProcessing);
        const nextImageTensor = images.next().value as Tensor3D

        if(nextImageTensor){
          const prediction = model.predict(tf.stack([nextImageTensor])) as tf.Tensor;
          nextImageTensor.dispose();

          const data = prediction.dataSync()
          prediction.dispose();

          const allPredictions = labels.map((label, index) =>{
            return {
              label,
              probability: data[index]
            }
          });

          const top3 = allPredictions.sort((a, b) => b.probability - a.probability).slice(0, 3);
          
          setTopPredictions(top3);
        }
      

        //
        // do something with tensor here
        //


        // if autorender is false you need the following two lines.
        // updatePreview();
        // gl.endFrameEXP();
      }
      
      requestAnimationFrame(loop);
    }
    loop();
  }, [])

  return (
    <View style={styles.container}>
      <TensorCamera 
        autorender
        onReady={handleCameraStream}
        cameraTextureHeight={width}
        cameraTextureWidth={width}
        resizeDepth={3}
        resizeHeight={299}
        resizeWidth={299}
        useCustomShadersToResize={false}
        onCameraReady={() => setCameraReady(true)} style={{ height: width, width }} type='back' />
      <Text>{ !model ? 'Loading model..' : hasPermission === false ? 'No permission to camera' : topPredictions ? JSON.stringify(topPredictions) : 'All good!' }</Text>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
