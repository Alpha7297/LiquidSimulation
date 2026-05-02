import taichi as ti

ti.init(arch=ti.cpu)

vec3=ti.types.vector(3,dtype=ti.f32)
ivec3=ti.types.vector(3,dtype=ti.i32)

PI=3.14159265358
G=-2.0
ETA=0.04
GRID_SIZE=18
DOMAIN_MIN=0.1
DOMAIN_MAX=0.9
DOMAIN_LEN=DOMAIN_MAX-DOMAIN_MIN
GRID_LEN=DOMAIN_LEN/float(GRID_SIZE)
LIQUID_CELL_X=7
LIQUID_CELL_Y=8
LIQUID_CELL_Z=7
LIQUID_START_I=3
LIQUID_START_J=2
LIQUID_START_K=3
PARTICLES_PER_CELL_AXIS=3
PARTICLES_PER_CELL=PARTICLES_PER_CELL_AXIS*PARTICLES_PER_CELL_AXIS*PARTICLES_PER_CELL_AXIS
PARTICLE_DX=GRID_LEN/float(PARTICLES_PER_CELL_AXIS)
MASS=0.00018/float(PARTICLES_PER_CELL)
H=1.6*PARTICLE_DX
DT_INIT=0.0015
DT_MIN=0.0003
DT_MAX=0.004
GAMMA=4.0
STIFFNESS=18.0
DX=PARTICLE_DX
MAX_NEIGHBOUR=512
SEARCH_GRID_SIZE=24
SEARCH_CELL_NUM=SEARCH_GRID_SIZE*SEARCH_GRID_SIZE*SEARCH_GRID_SIZE
SEARCH_MAX_BUCKET=128
SEARCH_MIN=0.0
SEARCH_MAX=1.0
SEARCH_GRID_LEN=(SEARCH_MAX-SEARCH_MIN)/float(SEARCH_GRID_SIZE)
SUBSTEPS=2
PART_RADIUS=0.0032
BOUNDARY_RADIUS=0.0018
BOUNDARY_RENDER_STRIDE=6
ACC_LIMIT=120.0
VEL_LIMIT=5.0
SOUND_SPEED=5.0
ART_VISC_ALPHA=0.06
ART_VISC_EPS=0.01
BOUNDARY_PRESSURE_SCALE=1.0
WALL_TANGENT_DAMP=0.96
BOX_LINE_VERTEX_NUM=24

FLUID_N=LIQUID_CELL_X*LIQUID_CELL_Y*LIQUID_CELL_Z*PARTICLES_PER_CELL
BOTTOM_LAYERS=3
SIDE_LAYERS=3
CONTAINER_N=(GRID_SIZE-2)*PARTICLES_PER_CELL_AXIS+1
SIDE_ROWS=CONTAINER_N
BOTTOM_COLS=CONTAINER_N+2*SIDE_LAYERS
BOTTOM_N=BOTTOM_COLS*BOTTOM_COLS*BOTTOM_LAYERS
SIDE_X_N=2*SIDE_ROWS*CONTAINER_N*SIDE_LAYERS
SIDE_Z_N=2*SIDE_ROWS*CONTAINER_N*SIDE_LAYERS
BOUNDARY_N=BOTTOM_N+SIDE_X_N+SIDE_Z_N
BOUNDARY_RENDER_N=(BOUNDARY_N+BOUNDARY_RENDER_STRIDE-1)//BOUNDARY_RENDER_STRIDE
PARTICLE_N=FLUID_N+BOUNDARY_N

WALL_MIN=DOMAIN_MIN+GRID_LEN
WALL_MAX=DOMAIN_MAX-GRID_LEN
X_LEFT=WALL_MIN
X_RIGHT=WALL_MAX
Y_BOTTOM=WALL_MIN
Y_TOP=WALL_MAX
Z_FRONT=WALL_MIN
Z_BACK=WALL_MAX
BOTTOM_X0=X_LEFT-SIDE_LAYERS*DX
BOTTOM_Z0=Z_FRONT-SIDE_LAYERS*DX
COLLISION_X_LEFT=X_LEFT+PART_RADIUS
COLLISION_X_RIGHT=X_RIGHT-PART_RADIUS
COLLISION_Y_BOTTOM=Y_BOTTOM+PART_RADIUS
COLLISION_Y_TOP=Y_TOP-PART_RADIUS
COLLISION_Z_FRONT=Z_FRONT+PART_RADIUS
COLLISION_Z_BACK=Z_BACK-PART_RADIUS

pos=ti.Vector.field(3,dtype=ti.f32,shape=PARTICLE_N)
vel=ti.Vector.field(3,dtype=ti.f32,shape=PARTICLE_N)
acc=ti.Vector.field(3,dtype=ti.f32,shape=PARTICLE_N)
rho=ti.field(dtype=ti.f32,shape=PARTICLE_N)
pressure=ti.field(dtype=ti.f32,shape=PARTICLE_N)
mass=ti.field(dtype=ti.f32,shape=PARTICLE_N)
edge=ti.field(dtype=ti.i32,shape=PARTICLE_N)
neighbour=ti.field(dtype=ti.i32,shape=PARTICLE_N*MAX_NEIGHBOUR)
neighbour_count=ti.field(dtype=ti.i32,shape=PARTICLE_N)
bucket_count=ti.field(dtype=ti.i32,shape=SEARCH_CELL_NUM)
bucket_particle=ti.field(dtype=ti.i32,shape=SEARCH_CELL_NUM*SEARCH_MAX_BUCKET)
rest_density=ti.field(dtype=ti.f32,shape=())
avg_neighbour_count=ti.field(dtype=ti.f32,shape=())
max_pressure=ti.field(dtype=ti.f32,shape=())
max_speed=ti.field(dtype=ti.f32,shape=())
acc_limit_count=ti.field(dtype=ti.i32,shape=())
vel_limit_count=ti.field(dtype=ti.i32,shape=())
dt=ti.field(dtype=ti.f32,shape=())

fluid_pos=ti.Vector.field(3,dtype=ti.f32,shape=FLUID_N)
boundary_pos=ti.Vector.field(3,dtype=ti.f32,shape=BOUNDARY_RENDER_N)
box_lines=ti.Vector.field(3,dtype=ti.f32,shape=BOX_LINE_VERTEX_NUM)

paused=False
show_boundary=True

@ti.func
def clamp_f(v:ti.f32,lo:ti.f32,hi:ti.f32)->ti.f32:
    return ti.min(ti.max(v,lo),hi)

@ti.func
def clamp_i(v:int,lo:int,hi:int)->int:
    return ti.min(ti.max(v,lo),hi)

@ti.func
def dist(a:vec3,b:vec3)->ti.f32:
    d=a-b
    return ti.sqrt(d.dot(d))

@ti.func
def search_cell_id(i:int,j:int,k:int)->int:
    return (i*SEARCH_GRID_SIZE+j)*SEARCH_GRID_SIZE+k

@ti.func
def search_cell_coord(p:vec3)->ivec3:
    i=clamp_i(ti.cast(ti.floor((p[0]-SEARCH_MIN)/SEARCH_GRID_LEN),ti.i32),0,SEARCH_GRID_SIZE-1)
    j=clamp_i(ti.cast(ti.floor((p[1]-SEARCH_MIN)/SEARCH_GRID_LEN),ti.i32),0,SEARCH_GRID_SIZE-1)
    k=clamp_i(ti.cast(ti.floor((p[2]-SEARCH_MIN)/SEARCH_GRID_LEN),ti.i32),0,SEARCH_GRID_SIZE-1)
    return ti.Vector([i,j,k])

@ti.func
def search_cell_id_from_pos(p:vec3)->int:
    c=search_cell_coord(p)
    return search_cell_id(c[0],c[1],c[2])

@ti.func
def W(r:ti.f32)->ti.f32:
    q=r/H
    res=0.0
    if q<=1.0:
        res=1.0-1.5*q*q+0.75*q*q*q
    elif q<=2.0:
        res=0.25*(2.0-q)*(2.0-q)*(2.0-q)
    return res/(PI*H*H*H)

@ti.func
def dW(r:ti.f32)->ti.f32:
    q=r/H
    res=0.0
    if q<=1.0:
        res=-3.0*q+2.25*q*q
    elif q<=2.0:
        res=-0.75*(2.0-q)*(2.0-q)
    return res/(PI*H*H*H*H)

@ti.kernel
def init_box_lines():
    p000=ti.Vector([X_LEFT,Y_BOTTOM,Z_FRONT])
    p100=ti.Vector([X_RIGHT,Y_BOTTOM,Z_FRONT])
    p010=ti.Vector([X_LEFT,Y_TOP,Z_FRONT])
    p110=ti.Vector([X_RIGHT,Y_TOP,Z_FRONT])
    p001=ti.Vector([X_LEFT,Y_BOTTOM,Z_BACK])
    p101=ti.Vector([X_RIGHT,Y_BOTTOM,Z_BACK])
    p011=ti.Vector([X_LEFT,Y_TOP,Z_BACK])
    p111=ti.Vector([X_RIGHT,Y_TOP,Z_BACK])
    box_lines[0]=p000
    box_lines[1]=p100
    box_lines[2]=p100
    box_lines[3]=p110
    box_lines[4]=p110
    box_lines[5]=p010
    box_lines[6]=p010
    box_lines[7]=p000
    box_lines[8]=p001
    box_lines[9]=p101
    box_lines[10]=p101
    box_lines[11]=p111
    box_lines[12]=p111
    box_lines[13]=p011
    box_lines[14]=p011
    box_lines[15]=p001
    box_lines[16]=p000
    box_lines[17]=p001
    box_lines[18]=p100
    box_lines[19]=p101
    box_lines[20]=p010
    box_lines[21]=p011
    box_lines[22]=p110
    box_lines[23]=p111

@ti.kernel
def init_particles():
    max_pressure[None]=1e-5
    max_speed[None]=1e-5
    acc_limit_count[None]=0
    vel_limit_count[None]=0
    for idx in range(FLUID_N):
        cell=idx//PARTICLES_PER_CELL
        local=idx%PARTICLES_PER_CELL
        cell_i=cell%LIQUID_CELL_X
        cell_j=(cell//LIQUID_CELL_X)%LIQUID_CELL_Y
        cell_k=cell//(LIQUID_CELL_X*LIQUID_CELL_Y)
        local_i=local%PARTICLES_PER_CELL_AXIS
        local_j=(local//PARTICLES_PER_CELL_AXIS)%PARTICLES_PER_CELL_AXIS
        local_k=local//(PARTICLES_PER_CELL_AXIS*PARTICLES_PER_CELL_AXIS)
        jitter_x=0.05*(ti.cast((idx*17+13)%29,ti.f32)/29.0-0.5)
        jitter_y=0.05*(ti.cast((idx*31+7)%37,ti.f32)/37.0-0.5)
        jitter_z=0.05*(ti.cast((idx*19+11)%31,ti.f32)/31.0-0.5)
        offset_x=(ti.cast(local_i,ti.f32)+0.5+jitter_x)/ti.cast(PARTICLES_PER_CELL_AXIS,ti.f32)
        offset_y=(ti.cast(local_j,ti.f32)+0.5+jitter_y)/ti.cast(PARTICLES_PER_CELL_AXIS,ti.f32)
        offset_z=(ti.cast(local_k,ti.f32)+0.5+jitter_z)/ti.cast(PARTICLES_PER_CELL_AXIS,ti.f32)
        pos[idx]=ti.Vector([
            DOMAIN_MIN+(ti.cast(LIQUID_START_I+cell_i,ti.f32)+offset_x)*GRID_LEN,
            DOMAIN_MIN+(ti.cast(LIQUID_START_J+cell_j,ti.f32)+offset_y)*GRID_LEN,
            DOMAIN_MIN+(ti.cast(LIQUID_START_K+cell_k,ti.f32)+offset_z)*GRID_LEN,
        ])
        vel[idx]=ti.Vector([0.0,0.0,0.0])
        acc[idx]=ti.Vector([0.0,0.0,0.0])
        rho[idx]=1.0
        pressure[idx]=0.0
        mass[idx]=MASS
        edge[idx]=0
        neighbour_count[idx]=0
    base=FLUID_N
    for i,k,l in ti.ndrange(BOTTOM_COLS,BOTTOM_COLS,BOTTOM_LAYERS):
        idx=base+(i*BOTTOM_COLS+k)*BOTTOM_LAYERS+l
        pos[idx]=ti.Vector([
            BOTTOM_X0+ti.cast(i,ti.f32)*DX,
            Y_BOTTOM-ti.cast(l+1,ti.f32)*DX,
            BOTTOM_Z0+ti.cast(k,ti.f32)*DX,
        ])
        vel[idx]=ti.Vector([0.0,0.0,0.0])
        acc[idx]=ti.Vector([0.0,0.0,0.0])
        rho[idx]=1.0
        pressure[idx]=0.0
        mass[idx]=MASS
        edge[idx]=1
        neighbour_count[idx]=0
    base=FLUID_N+BOTTOM_N
    for y,z,l in ti.ndrange(SIDE_ROWS,CONTAINER_N,SIDE_LAYERS):
        local=(y*CONTAINER_N+z)*SIDE_LAYERS+l
        left=base+local
        right=base+SIDE_ROWS*CONTAINER_N*SIDE_LAYERS+local
        yy=Y_BOTTOM+ti.cast(y,ti.f32)*DX
        zz=Z_FRONT+ti.cast(z,ti.f32)*DX
        pos[left]=ti.Vector([X_LEFT-ti.cast(l+1,ti.f32)*DX,yy,zz])
        pos[right]=ti.Vector([X_RIGHT+ti.cast(l+1,ti.f32)*DX,yy,zz])
        vel[left]=ti.Vector([0.0,0.0,0.0])
        vel[right]=ti.Vector([0.0,0.0,0.0])
        acc[left]=ti.Vector([0.0,0.0,0.0])
        acc[right]=ti.Vector([0.0,0.0,0.0])
        rho[left]=1.0
        rho[right]=1.0
        pressure[left]=0.0
        pressure[right]=0.0
        mass[left]=MASS
        mass[right]=MASS
        edge[left]=1
        edge[right]=1
        neighbour_count[left]=0
        neighbour_count[right]=0
    base=FLUID_N+BOTTOM_N+SIDE_X_N
    for y,x,l in ti.ndrange(SIDE_ROWS,CONTAINER_N,SIDE_LAYERS):
        local=(y*CONTAINER_N+x)*SIDE_LAYERS+l
        front=base+local
        back=base+SIDE_ROWS*CONTAINER_N*SIDE_LAYERS+local
        yy=Y_BOTTOM+ti.cast(y,ti.f32)*DX
        xx=X_LEFT+ti.cast(x,ti.f32)*DX
        pos[front]=ti.Vector([xx,yy,Z_FRONT-ti.cast(l+1,ti.f32)*DX])
        pos[back]=ti.Vector([xx,yy,Z_BACK+ti.cast(l+1,ti.f32)*DX])
        vel[front]=ti.Vector([0.0,0.0,0.0])
        vel[back]=ti.Vector([0.0,0.0,0.0])
        acc[front]=ti.Vector([0.0,0.0,0.0])
        acc[back]=ti.Vector([0.0,0.0,0.0])
        rho[front]=1.0
        rho[back]=1.0
        pressure[front]=0.0
        pressure[back]=0.0
        mass[front]=MASS
        mass[back]=MASS
        edge[front]=1
        edge[back]=1
        neighbour_count[front]=0
        neighbour_count[back]=0

@ti.kernel
def find_neighbours():
    for i in range(SEARCH_CELL_NUM):
        bucket_count[i]=0
    for i in range(PARTICLE_N):
        gid=search_cell_id_from_pos(pos[i])
        slot=ti.atomic_add(bucket_count[gid],1)
        if slot<SEARCH_MAX_BUCKET:
            bucket_particle[gid*SEARCH_MAX_BUCKET+slot]=i
    total=0
    for i in range(PARTICLE_N):
        count=0
        cell=search_cell_coord(pos[i])
        for di,dj,dk in ti.ndrange((-2,3),(-2,3),(-2,3)):
            ni=cell[0]+di
            nj=cell[1]+dj
            nk=cell[2]+dk
            if ni>=0 and ni<SEARCH_GRID_SIZE and nj>=0 and nj<SEARCH_GRID_SIZE and nk>=0 and nk<SEARCH_GRID_SIZE:
                gid=search_cell_id(ni,nj,nk)
                cell_count=ti.min(bucket_count[gid],SEARCH_MAX_BUCKET)
                for s in range(cell_count):
                    j=bucket_particle[gid*SEARCH_MAX_BUCKET+s]
                    if i!=j:
                        r=dist(pos[i],pos[j])
                        if r<2.0*H and count<MAX_NEIGHBOUR:
                            neighbour[i*MAX_NEIGHBOUR+count]=j
                            count+=1
        neighbour_count[i]=count
        total+=count
    avg_neighbour_count[None]=ti.cast(total,ti.f32)/ti.cast(PARTICLE_N,ti.f32)

@ti.kernel
def compute_density():
    for i in range(PARTICLE_N):
        value=mass[i]*W(0.0)
        for j in range(neighbour_count[i]):
            nb=neighbour[i*MAX_NEIGHBOUR+j]
            r=dist(pos[i],pos[nb])
            value+=mass[nb]*W(r)
        rho[i]=ti.max(value,1e-4)

@ti.kernel
def init_rest_density():
    total=0.0
    for i in range(FLUID_N):
        total+=rho[i]
    rest_density[None]=total/ti.cast(FLUID_N,ti.f32)

@ti.kernel
def compute_pressure():
    max_pressure[None]=1e-4
    for i in range(PARTICLE_N):
        ratio=rho[i]/rest_density[None]
        raw=ti.max(0.0,STIFFNESS/GAMMA*(ti.pow(ratio,GAMMA)-1.0))
        pressure[i]=raw
        if edge[i]==1:
            pressure[i]=BOUNDARY_PRESSURE_SCALE*raw
        if edge[i]==0:
            ti.atomic_max(max_pressure[None],pressure[i])

@ti.kernel
def compute_acceleration():
    acc_limit_count[None]=0
    for i in range(PARTICLE_N):
        acc[i]=ti.Vector([0.0,0.0,0.0])
        if edge[i]==0:
            a=ti.Vector([0.0,G,0.0])
            for j in range(neighbour_count[i]):
                nb=neighbour[i*MAX_NEIGHBOUR+j]
                rij=pos[i]-pos[nb]
                r=ti.sqrt(rij.dot(rij))+1e-5
                if r<2.0*H:
                    grad=rij/r*dW(r)
                    pressure_term=pressure[i]/(rho[i]*rho[i])+pressure[nb]/(rho[nb]*rho[nb])
                    a+=-mass[nb]*pressure_term*grad
                    if edge[nb]==0:
                        vij=vel[i]-vel[nb]
                        approach=vij.dot(rij)
                        if approach<0.0:
                            rho_bar=0.5*(rho[i]+rho[nb])
                            mu=H*approach/(r*r+ART_VISC_EPS*H*H)
                            pi_ij=-ART_VISC_ALPHA*SOUND_SPEED*mu/rho_bar
                            a+=-mass[nb]*pi_ij*grad
                        a+=ETA*mass[nb]*(vel[nb]-vel[i])*W(r)/rho[nb]
            norm=a.norm()
            if norm>ACC_LIMIT:
                a=a/norm*ACC_LIMIT
                ti.atomic_add(acc_limit_count[None],1)
            acc[i]=a

@ti.kernel
def integrate():
    max_speed[None]=1e-5
    vel_limit_count[None]=0
    for i in range(PARTICLE_N):
        if edge[i]==0:
            vel[i]+=dt[None]*acc[i]
            speed=vel[i].norm()
            if speed>VEL_LIMIT:
                vel[i]=vel[i]/speed*VEL_LIMIT
                speed=VEL_LIMIT
                ti.atomic_add(vel_limit_count[None],1)
            pos[i]+=dt[None]*vel[i]
            if pos[i][0]<COLLISION_X_LEFT:
                pos[i][0]=COLLISION_X_LEFT
                if vel[i][0]<0.0:
                    vel[i][0]=0.0
                vel[i][1]*=WALL_TANGENT_DAMP
                vel[i][2]*=WALL_TANGENT_DAMP
            if pos[i][0]>COLLISION_X_RIGHT:
                pos[i][0]=COLLISION_X_RIGHT
                if vel[i][0]>0.0:
                    vel[i][0]=0.0
                vel[i][1]*=WALL_TANGENT_DAMP
                vel[i][2]*=WALL_TANGENT_DAMP
            if pos[i][1]<COLLISION_Y_BOTTOM:
                pos[i][1]=COLLISION_Y_BOTTOM
                if vel[i][1]<0.0:
                    vel[i][1]=0.0
                vel[i][0]*=WALL_TANGENT_DAMP
                vel[i][2]*=WALL_TANGENT_DAMP
            if pos[i][1]>COLLISION_Y_TOP:
                pos[i][1]=COLLISION_Y_TOP
                if vel[i][1]>0.0:
                    vel[i][1]=0.0
                vel[i][0]*=WALL_TANGENT_DAMP
                vel[i][2]*=WALL_TANGENT_DAMP
            if pos[i][2]<COLLISION_Z_FRONT:
                pos[i][2]=COLLISION_Z_FRONT
                if vel[i][2]<0.0:
                    vel[i][2]=0.0
                vel[i][0]*=WALL_TANGENT_DAMP
                vel[i][1]*=WALL_TANGENT_DAMP
            if pos[i][2]>COLLISION_Z_BACK:
                pos[i][2]=COLLISION_Z_BACK
                if vel[i][2]>0.0:
                    vel[i][2]=0.0
                vel[i][0]*=WALL_TANGENT_DAMP
                vel[i][1]*=WALL_TANGENT_DAMP
            ti.atomic_max(max_speed[None],speed)
        else:
            vel[i]=ti.Vector([0.0,0.0,0.0])

@ti.kernel
def update_render_fields():
    for i in range(FLUID_N):
        fluid_pos[i]=pos[i]
    for i in range(BOUNDARY_RENDER_N):
        boundary_pos[i]=pos[FLUID_N+i*BOUNDARY_RENDER_STRIDE]

def substep():
    find_neighbours()
    compute_density()
    compute_pressure()
    compute_acceleration()
    integrate()

def init():
    dt[None]=DT_INIT
    init_box_lines()
    init_particles()
    find_neighbours()
    compute_density()
    init_rest_density()
    compute_pressure()
    update_render_fields()

def init_camera(camera):
    camera.position(1.45,1.15,1.65)
    camera.lookat(0.50,0.45,0.50)
    camera.up(0.0,1.0,0.0)
    camera.fov(55)

def clamp(v,lo,hi):
    return max(lo,min(v,hi))

def handle_input(window,camera):
    global paused,show_boundary
    for e in window.get_events(ti.ui.PRESS):
        if e.key==ti.ui.ESCAPE:
            window.running=False
        elif e.key==" ":
            paused=not paused
        elif e.key=="r" or e.key=="R":
            init()
        elif e.key=="c" or e.key=="C":
            init_camera(camera)
        elif e.key=="b" or e.key=="B":
            show_boundary=not show_boundary
        elif e.key=="[":
            dt[None]=clamp(dt[None]*0.8,DT_MIN,DT_MAX)
        elif e.key=="]":
            dt[None]=clamp(dt[None]*1.25,DT_MIN,DT_MAX)

def render(window,scene,camera):
    update_render_fields()
    scene.set_camera(camera)
    scene.ambient_light((0.35,0.35,0.35))
    scene.point_light(pos=(1.4,1.6,1.2),color=(1.0,1.0,1.0))
    scene.lines(box_lines,width=1.5,color=(0.45,0.48,0.52))
    if show_boundary:
        scene.particles(boundary_pos,radius=BOUNDARY_RADIUS,color=(0.62,0.68,0.74))
    scene.particles(fluid_pos,radius=PART_RADIUS,color=(0.18,0.52,0.95))
    canvas=window.get_canvas()
    canvas.set_background_color((0.07,0.08,0.09))
    canvas.scene(scene)

def render_gui(window):
    gui=window.get_gui()
    gui.begin("3D SPH",0.02,0.02,0.27,0.23)
    gui.text(f"particles: {PARTICLE_N}")
    gui.text(f"fluid: {FLUID_N}")
    gui.text(f"boundary shown: {BOUNDARY_RENDER_N}/{BOUNDARY_N}")
    gui.text(f"rest rho: {rest_density[None]:.3f}")
    gui.text(f"avg nb: {avg_neighbour_count[None]:.1f}")
    dt[None]=clamp(gui.slider_float("dt",dt[None],DT_MIN,DT_MAX),DT_MIN,DT_MAX)
    gui.text(f"stiffness: {STIFFNESS:.1f}")
    gui.text(f"max pressure: {max_pressure[None]:.3f}")
    gui.text(f"max speed: {max_speed[None]:.3f}")
    gui.text(f"acc cap: {acc_limit_count[None]}")
    gui.text(f"vel cap: {vel_limit_count[None]}")
    gui.end()

def main():
    init()
    window=ti.ui.Window("3D WCSPH",(1100,900))
    scene=ti.ui.Scene()
    camera=ti.ui.Camera()
    init_camera(camera)
    while window.running:
        handle_input(window,camera)
        camera.track_user_inputs(window,movement_speed=0.035,hold_key=ti.ui.RMB)
        if not paused:
            for _ in range(SUBSTEPS):
                substep()
        render(window,scene,camera)
        render_gui(window)
        window.show()

if __name__=="__main__":
    main()
