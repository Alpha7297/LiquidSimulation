import taichi as ti 
import numpy as np
ti.init(arch=ti.cpu)
GRID_SIZE=10
NUM_PARTICLES=GRID_SIZE*GRID_SIZE*GRID_SIZE
DT=0.01
parts_pos=ti.Vector.field(2,dtype=ti.f32,shape=NUM_PARTICLES)
parts_vel=ti.Vector.field(2,dtype=ti.f32,shape=NUM_PARTICLES)
def init():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            parts_pos[i*GRID_SIZE+j]=ti.Vector([i*0.01+0.2,j*0.01+0.2])
            parts_vel[i*GRID_SIZE+j]=ti.Vector([0.1,0])
def render(canvas:ti.ui.Canvas):
    canvas.set_background_color((0.1,0.1,0.1))
    canvas.circles(parts_pos,0.005,color=(0.5,0.5,0.5))
@ti.kernel
def update_pos(t:ti.f32):
    for i in range(NUM_PARTICLES):
        parts_vel[i]=0.1*ti.Vector([ti.cos(t),0.0])
        parts_pos[i]=parts_vel[i]*DT+parts_pos[i]
def main():
    init()
    window=ti.ui.Window("2D Particles",(1500,1500))
    canvas=window.get_canvas()
    t=0
    while window.running:
        update_pos(t)
        t+=DT
        render(canvas)
        window.show()

if __name__=="__main__":
    main()