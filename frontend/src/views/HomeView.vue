<script setup>
import UploadImage from '@/components/UploadImage.vue'
import { ref, onMounted } from 'vue'
import FooterBar from '@/components/FooterBar.vue'

import * as ort from 'onnxruntime-web'

// Set the path to the wasm files that are located in the public folder.
// This is required because vite does not handle the wasm files correctly.
// Interested in any better solution to this problem.

ort.env.wasm.wasmPaths = {
  'ort-wasm.wasm': './ort-wasm.wasm',
  'ort-wasm-simd.wasm': './ort-wasm-simd.wasm',
  'ort-wasm-threaded.wasm': './ort-wasm-threaded.wasm'
}

const uploadedImageSrc = ref(null)
const predictedImageSrc = ref(null)

async function handleImageUpload(file) {
  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader()
    reader.onload = async (e) => {
      uploadedImageSrc.value = e.target.result
    }
    reader.readAsDataURL(file)
  }
}

let session = null

onMounted(async () => {
  session = await ort.InferenceSession.create('./model2.onnx')
})

async function predict() {
  if (!session || !uploadedImageSrc.value) {
    console.error('Session not initialized or no image uploaded')
    return
  }

  const image = new Image()
  image.onload = async () => {
    const width = image.naturalWidth
    const height = image.naturalHeight

    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')
    ctx.drawImage(image, 0, 0, width, height)

    const imageData = ctx.getImageData(0, 0, width, height)

    // Convert imageData.data (Uint8ClampedArray) to Float32Array
    const float32Data = new Float32Array(width * height * 3)
    for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 3) {
      // Assuming the model expects RGB format
      float32Data[j] = imageData.data[i] / 255.0 // R
      float32Data[j + 1] = imageData.data[i + 1] / 255.0 // G
      float32Data[j + 2] = imageData.data[i + 2] / 255.0 // B
      // Skip the alpha channel
    }

    // Adjust the tensor shape and type according to your model's requirements
    const tensor = new ort.Tensor('float32', float32Data, [1, 3, height, width])

    let output = await session.run({ input: tensor })

    const outputTensor = output.output // Adjust based on your model's output
    const outputData = outputTensor.data
    // Create a new array for RGBA data
    const outputImageData = new Uint8ClampedArray(width * height * 4)
    for (let i = 0, j = 0; i < outputData.length; i += 3, j += 4) {
      outputImageData[j] = Math.min(255, Math.max(0, outputData[i] * 255)) // Red
      outputImageData[j + 1] = Math.min(255, Math.max(0, outputData[i + 1] * 255)) // Green
      outputImageData[j + 2] = Math.min(255, Math.max(0, outputData[i + 2] * 255)) // Blue
      outputImageData[j + 3] = 255 // Alpha channel (fully opaque)
    }
    const outputImage = new ImageData(outputImageData, width, height)
    ctx.putImageData(outputImage, 0, 0)

    predictedImageSrc.value = canvas.toDataURL()
  }
  image.onerror = () => {
    console.error('Error loading the image')
  }
  image.src = uploadedImageSrc.value
}
</script>

<template>
  <main class="h-screen w-full flex flex-col justify-between items-center">
    <UploadImage @upload="handleImageUpload" />
    <img class="max-w-[400px]" v-if="uploadedImageSrc" :src="uploadedImageSrc" />
    <button
      v-if="uploadedImageSrc"
      class="bg-blue-500 mt-10 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
      @click="predict"
    >
      Predict
    </button>
    <img class="max-w-[400px] mt-5" v-if="predictedImageSrc" :src="predictedImageSrc" />

    <FooterBar class="w-full"></FooterBar>
  </main>
</template>
