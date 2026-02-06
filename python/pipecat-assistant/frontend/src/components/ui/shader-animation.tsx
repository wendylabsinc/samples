import { useEffect, useRef, useImperativeHandle, forwardRef } from "react"
import * as THREE from "three"

export interface ShaderAnimationRef {
  setAudioLevel: (level: number) => void
}

interface ShaderAnimationProps {
  audioLevel?: number
}

export const ShaderAnimation = forwardRef<ShaderAnimationRef, ShaderAnimationProps>(
  function ShaderAnimation({ audioLevel = 0 }, ref) {
    const containerRef = useRef<HTMLDivElement>(null)
    const uniformsRef = useRef<{
      time: { value: number }
      resolution: { value: THREE.Vector2 }
      audioLevel: { value: number }
    } | null>(null)
    const sceneRef = useRef<{
      camera: THREE.Camera
      scene: THREE.Scene
      renderer: THREE.WebGLRenderer
      animationId: number
    } | null>(null)

    // Expose method to update audio level
    useImperativeHandle(ref, () => ({
      setAudioLevel: (level: number) => {
        if (uniformsRef.current) {
          uniformsRef.current.audioLevel.value = level
        }
      },
    }))

    // Update audio level from prop
    useEffect(() => {
      if (uniformsRef.current) {
        uniformsRef.current.audioLevel.value = audioLevel
      }
    }, [audioLevel])

    useEffect(() => {
      if (!containerRef.current) return

      const container = containerRef.current

      // Vertex shader
      const vertexShader = `
        void main() {
          gl_Position = vec4( position, 1.0 );
        }
      `

      // Fragment shader - pulsing orb effect that reacts to audio
      const fragmentShader = `
        #define TWO_PI 6.2831853072
        #define PI 3.14159265359

        precision highp float;
        uniform vec2 resolution;
        uniform float time;
        uniform float audioLevel;

        void main(void) {
          vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / min(resolution.x, resolution.y);
          float t = time * 0.03;

          // Audio reactivity - smoothed level for visual effect
          float audio = audioLevel;
          float audioBoost = 1.0 + audio * 2.0;

          // Create pulsing orb effect - pulse is enhanced by audio
          float dist = length(uv);
          float basePulse = sin(t * 3.0) * 0.1 + 0.9;
          float pulse = basePulse + audio * 0.3;

          // Multiple layered rings - intensity and movement affected by audio
          vec3 color = vec3(0.0);
          for(int i = 0; i < 5; i++){
            float fi = float(i);
            // Rings expand with audio
            float radius = (0.25 + fi * 0.15) * (1.0 + audio * 0.5);
            // Ring thickness increases with audio
            float thickness = 0.02 + audio * 0.015;
            float ring = smoothstep(thickness, 0.0, abs(dist - radius * pulse + sin(t + fi) * 0.05 * audioBoost));
            // Color intensity increases with audio
            float intensity = 1.0 + audio * 1.5;
            color += ring * vec3(0.2 + fi * 0.1, 0.4 + fi * 0.1, 0.8) * intensity;
          }

          // Center glow - brighter with audio
          float glowIntensity = 0.15 + audio * 0.3;
          float glow = glowIntensity / (dist + 0.1);
          color += glow * vec3(0.3, 0.5, 1.0) * (0.3 + audio * 0.5);

          // Ambient pulse - more vibrant with audio
          float ambientIntensity = sin(t * 2.0) * 0.3 + 0.7 + audio * 0.3;
          color += vec3(0.05, 0.1, 0.2) * ambientIntensity;

          // Add slight color shift with high audio levels
          if (audio > 0.5) {
            color.r += (audio - 0.5) * 0.3;
          }

          gl_FragColor = vec4(color, 1.0);
        }
      `

      // Initialize Three.js scene
      const camera = new THREE.Camera()
      camera.position.z = 1

      const scene = new THREE.Scene()
      const geometry = new THREE.PlaneGeometry(2, 2)

      const uniforms = {
        time: { value: 1.0 },
        resolution: { value: new THREE.Vector2() },
        audioLevel: { value: 0.0 },
      }
      uniformsRef.current = uniforms

      const material = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
      })

      const mesh = new THREE.Mesh(geometry, material)
      scene.add(mesh)

      const renderer = new THREE.WebGLRenderer({ antialias: true })
      renderer.setPixelRatio(window.devicePixelRatio)

      container.appendChild(renderer.domElement)

      // Handle window resize
      const onWindowResize = () => {
        const width = container.clientWidth
        const height = container.clientHeight
        renderer.setSize(width, height)
        uniforms.resolution.value.x = renderer.domElement.width
        uniforms.resolution.value.y = renderer.domElement.height
      }

      // Initial resize
      onWindowResize()
      window.addEventListener("resize", onWindowResize, false)

      // Animation loop
      const animate = () => {
        const animationId = requestAnimationFrame(animate)
        uniforms.time.value += 0.05
        renderer.render(scene, camera)

        if (sceneRef.current) {
          sceneRef.current.animationId = animationId
        }
      }

      // Store scene references for cleanup
      sceneRef.current = {
        camera,
        scene,
        renderer,
        animationId: 0,
      }

      // Start animation
      animate()

      // Cleanup function
      return () => {
        window.removeEventListener("resize", onWindowResize)

        if (sceneRef.current) {
          cancelAnimationFrame(sceneRef.current.animationId)

          if (container && sceneRef.current.renderer.domElement) {
            container.removeChild(sceneRef.current.renderer.domElement)
          }

          sceneRef.current.renderer.dispose()
          geometry.dispose()
          material.dispose()
        }
      }
    }, [])

    return (
      <div
        ref={containerRef}
        className="w-full h-screen"
        style={{
          background: "#000",
          overflow: "hidden",
        }}
      />
    )
  }
)
