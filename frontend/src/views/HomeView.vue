<script setup>
import { ref, onMounted } from 'vue'

import UploadImage from '@/components/UploadImage.vue'
import FooterBar from '@/components/FooterBar.vue'
import IconDownload from '@/components/icons/IconDownload.vue'
import IconTrash from '@/components/icons/IconTrash.vue'
import IconEye from '@/components/icons/IconEye.vue'
import IconEyeClosed from '@/components/icons/IconEyeClosed.vue'
import HeroSection from '@/components/HeroSection.vue'

import { loadImage, getImageData, normalizeAndTranspose, transformAndTranspose } from '@/utils.js'

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
const activeImageSrc = ref(null)

async function handleImageUpload(file) {
  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader()
    reader.onload = async (e) => {
      uploadedImageSrc.value = e.target.result
      activeImageSrc.value = e.target.result
    }
    reader.readAsDataURL(file)
  }
}

let session = null

async function predict() {
  if (!session || !uploadedImageSrc.value) {
    console.error('Session not initialized or no image uploaded')
    return
  }

  const image = await loadImage(uploadedImageSrc.value)

  const imageData = getImageData(image)
  const float32Data = normalizeAndTranspose(imageData, image.width, image.height)

  const tensor = new ort.Tensor('float32', float32Data, [1, 3, image.height, image.width])

  const output = await session.run({ input: tensor })
  const outputImageData = transformAndTranspose(output.output, image.width, image.height)

  const outputImage = new ImageData(outputImageData, image.width, image.height)
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  canvas.width = image.width
  canvas.height = image.height
  ctx.putImageData(outputImage, 0, 0)

  predictedImageSrc.value = canvas.toDataURL()
  activeImageSrc.value = canvas.toDataURL()
}

function reset() {
  uploadedImageSrc.value = null
  predictedImageSrc.value = null
  activeImageSrc.value = null
}

function switchImgSrc() {
  if (activeImageSrc.value == uploadedImageSrc.value) {
    activeImageSrc.value = predictedImageSrc.value
  } else {
    activeImageSrc.value = uploadedImageSrc.value
  }
}

function download() {
  const link = document.createElement('a')
  link.href = activeImageSrc.value
  link.download = 'image.png'
  link.click()
}

onMounted(async () => {
  const executionProviders = ['wasm']
  session = await ort.InferenceSession.create('./model.onnx', { executionProviders })
})
</script>

<template>
  <main class="h-screen w-full flex flex-col justify-between items-center">
    <div class="flex flex-col justify-start items-center">
      <HeroSection>
        <template v-slot:title> Transform Night into Light </template>
        <template v-slot:description>
          Transform your dark photos into bright images directly in your browser. No server uploads,
          ensuring complete privacy.
        </template>
      </HeroSection>
      <UploadImage v-if="!uploadedImageSrc" @upload="handleImageUpload" />
      <div v-else class="flex flex-row justify-center mb-2 items-center">
        <button
          v-if="predictedImageSrc"
          class="text-gray-500 hover:text-gray-900"
          @click="switchImgSrc()"
        >
          <IconEye v-if="activeImageSrc === predictedImageSrc" />
          <IconEyeClosed v-else />
        </button>
        <button
          class="text-gray-500 hover:text-gray-900 mx-3 font-bold bg-gray-200 rounded px-2"
          @click="predict"
        >
          Transform
        </button>
        <button
          v-if="predictedImageSrc"
          class="mr-3 text-gray-500 hover:text-gray-900"
          @click="download()"
        >
          <IconDownload />
        </button>
        <button class="text-gray-500 hover:text-gray-900" @click="reset()">
          <IconTrash />
        </button>
      </div>
      <img
        class="max-w-[330px] lg:max-w-[500px] max-h-[400px] lg:max-h-[500px]"
        v-if="activeImageSrc"
        :src="activeImageSrc"
      />
    </div>
    <FooterBar class="w-full"></FooterBar>
  </main>
</template>
