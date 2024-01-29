export async function loadImage(url) {
  const response = await fetch(url)
  const blob = await response.blob()
  return createImageBitmap(blob)
}

// Function to get ImageData from an ImageBitmap
export function getImageData(image) {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  canvas.width = image.width
  canvas.height = image.height
  ctx.drawImage(image, 0, 0)
  return ctx.getImageData(0, 0, canvas.width, canvas.height)
}

export function normalizeAndTranspose(imageData, width, height) {
  const float32Data = new Float32Array(width * height * 3)
  const rArray = new Float32Array(width * height)
  const gArray = new Float32Array(width * height)
  const bArray = new Float32Array(width * height)

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4
      const i = y * width + x
      rArray[i] = imageData.data[idx] / 255
      gArray[i] = imageData.data[idx + 1] / 255
      bArray[i] = imageData.data[idx + 2] / 255
    }
  }

  float32Data.set(rArray)
  float32Data.set(gArray, rArray.length)
  float32Data.set(bArray, rArray.length + gArray.length)
  return float32Data
}

export function transformAndTranspose(outputTensor, width, height, channels=3) {
  const transposedData = new Float32Array(width * height * channels)
  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      for (let c = 0; c < channels; c++) {
        transposedData[h * width * channels + w * channels + c] =
          outputTensor.data[c * height * width + h * width + w]
      }
    }
  }

  // Scale to [0, 255] and create Uint8ClampedArray for canvas
  const outimageData = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < transposedData.length; i += channels) {
    for (let c = 0; c < channels; c++) {
      outimageData[(i * 4) / 3 + c] = Math.min(
        255,
        Math.max(0, Math.round(transposedData[i + c] * 255))
      )
    }
    outimageData[(i * 4) / 3 + 3] = 255 // Alpha channel
  }
  return outimageData
}
