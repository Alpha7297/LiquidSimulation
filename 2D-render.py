import taichi as ti 
import numpy as np
ti.init(arch=ti.cpu)
vec2=ti.types.vector(2,dtype=ti.f32)

GRID_SIZE=10
GRID_LEN=0.8/float(GRID_SIZE)
NUM_PARTICLES=500
DT=0.01
ALPHA=0.95
#求解区域: 左右和上下均为0.1到0.9
#列优先存储，即第一个编号是列，第二个编号是行
parts_pos=ti.Vector.field(2,dtype=ti.f32,shape=NUM_PARTICLES)
old_parts_vel=ti.Vector.field(2,dtype=ti.f32,shape=NUM_PARTICLES)
parts_vel=ti.Vector.field(2,dtype=ti.f32,shape=NUM_PARTICLES)
new_part_vel=ti.Vector.field(2,dtype=ti.f32,shape=NUM_PARTICLES)
parts_grid=ti.field(dtype=ti.i32,shape=NUM_PARTICLES)

grid_pos=ti.Vector.field(2,dtype=ti.f32,shape=GRID_SIZE*GRID_SIZE)
grid_type=ti.field(dtype=ti.i32,shape=GRID_SIZE*GRID_SIZE)
grid_velx=ti.field(dtype=ti.f32,shape=GRID_SIZE*GRID_SIZE)
grid_vely=ti.field(dtype=ti.f32,shape=GRID_SIZE*GRID_SIZE)
grid_part_weight=ti.field(dtype=ti.f32,shape=GRID_SIZE*GRID_SIZE)
grid_cnt=ti.field(dtype=ti.i32,shape=GRID_SIZE*GRID_SIZE)
old_grid_vel_x=ti.field(dtype=ti.f32,shape=GRID_SIZE*GRID_SIZE)
old_grid_vel_y=ti.field(dtype=ti.f32,shape=GRID_SIZE*GRID_SIZE)
grid_pres=ti.field(dtype=ti.f32,shape=GRID_SIZE*GRID_SIZE)
@ti.func
def cal_grid_pos(i:int,j:int)->vec2:
    return ti.Vector([0.1+GRID_LEN*i,0.1+GRID_LEN*j])
@ti.func
def kern(r):
    return ti.max(1.0-ti.abs(r),0.0)
@ti.func
def weight(dr):
    norm_r=dr/GRID_LEN
    return kern(norm_r[0])*kern(norm_r[1])
@ti.kernel
def G2P():
    for i in range(NUM_PARTICLES):
        grid_id=parts_grid[i]
        ii=grid_id//GRID_SIZE#列
        jj=grid_id%GRID_SIZE#行
        ori_pos=cal_grid_pos(ii,jj)#左下角坐标
        pos=parts_pos[i]
        velx=parts_vel[i][0]
        vely=parts_vel[i][1]
        pos1=ori_pos+ti.Vector([GRID_LEN,GRID_LEN/2.0])
        pos2=ori_pos+ti.Vector([0,GRID_LEN/2.0])
        pos3=ori_pos+ti.Vector([GRID_LEN/2.0,0])
        pos4=ori_pos+ti.Vector([GRID_LEN/2.0,GRID_LEN])
        PIC_vx=(grid_velx[grid_id]*weight(pos1-pos)+grid_velx[grid_id-1]*weight(pos2-pos))/(weight(pos1-pos)+weight(pos2-pos))
        PIC_vy=(grid_vely[grid_id]*weight(pos3-pos)+grid_vely[grid_id+GRID_SIZE]*weight(pos4-pos))/(weight(pos3-pos)+weight(pos4-pos))
        FLIP_vx=old_parts_vel[i][0]+((grid_velx[grid_id]-old_grid_vel_x[grid_id])*weight(pos1-pos)+(grid_velx[grid_id-1]-old_grid_vel_x[grid_id-1])*weight(pos2-pos))/(weight(pos1-pos)+weight(pos2-pos))
        FLIP_vy=old_parts_vel[i][1]+((grid_vely[grid_id]-old_grid_vel_y[grid_id])*weight(pos3-pos)+(grid_vely[grid_id+GRID_SIZE]-old_grid_vel_y[grid_id+GRID_SIZE])*weight(pos4-pos))/(weight(pos3-pos)+weight(pos4-pos))
        new_grid_vel_x=(1-ALPHA)*PIC_vx+ALPHA*FLIP_vx
        new_grid_vel_y=(1-ALPHA)*PIC_vy+ALPHA*FLIP_vy
        new_part_vel[i]=ti.Vector([new_grid_vel_x,new_grid_vel_y])
@ti.kernel
def P2G():
    for i in range(GRID_SIZE*GRID_SIZE):
        old_grid_vel_x[i]=grid_velx[i]
        grid_part_weight[i]=0.0
    for i in range(NUM_PARTICLES):
        old_parts_vel[i]=parts_vel[i]
    for i in range(NUM_PARTICLES):
        
@ti.kernel
def update_part_grid_bind():
    for i in range(GRID_SIZE*GRID_SIZE):
        grid_cnt[i]=0
        
def init():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            parts_pos[i*GRID_SIZE+j]=ti.Vector([i*0.01+0.1,j*0.01+0.1])
            parts_vel[i*GRID_SIZE+j]=ti.Vector([0,0])
            
def render(canvas:ti.ui.Canvas):
    canvas.set_background_color((0.1,0.1,0.1))
    canvas.circles(parts_pos,0.005,color=(0.5,0.5,0.5))
@ti.kernel
def update_pos(t:ti.f32):
    for i in range(NUM_PARTICLES):
        parts_vel[i]=0.1*ti.Vector([ti.sin(t),0.0])
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