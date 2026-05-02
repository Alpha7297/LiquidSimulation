import taichi as ti

ti.init(arch=ti.cpu)

vec2=ti.types.vector(2,dtype=ti.f32)
vec3=ti.types.vector(3,dtype=ti.f32)

PI=3.14159265358
G=-2
ETA=0.08
MASS=0.0102
H=0.072
DT_INIT=0.003
DX=0.05
MAX_NEIGHBOUR=128
PRESSURE_ITERS=120
PRESSURE_RELAX=0.82
SUBSTEPS=2
PART_RADIUS=0.0032
BOUNDARY_RADIUS=0.0025
ACC_LIMIT=120.0
VEL_LIMIT=6.0
SOUND_SPEED=6.0
ART_VISC_ALPHA=0.06
ART_VISC_EPS=0.01
POISSON_EPS=1e-6
BOUNDARY_PRESSURE_SCALE=0.65
BOUNDARY_PRESSURE_RELAX=0.28
BOUNDARY_GRAD_SCALE=0.55
PRESSURE_COLOR_MIN=25.0
PRESSURE_COLOR_SMOOTH=0.96
VISUAL_PRESSURE_SMOOTH=0.88
WALL_TANGENT_DAMP=0.68
WALL_CONTACT_EPS=1e-4

FLUID_SIDE=29
FLUID_N=FLUID_SIDE*FLUID_SIDE
BOTTOM_LAYERS=3
SIDE_ROWS=70
SIDE_LAYERS=3
CONTAINER_COLS=56
BOTTOM_COLS=CONTAINER_COLS+2*SIDE_LAYERS
BOTTOM_N=BOTTOM_COLS*BOTTOM_LAYERS
SIDE_N=2*SIDE_ROWS*SIDE_LAYERS
BOUNDARY_N=BOTTOM_N+SIDE_N
PARTICLE_N=FLUID_N+BOUNDARY_N

X_LEFT=0.58
X_RIGHT=X_LEFT+(CONTAINER_COLS-1)*DX
Y_BOTTOM=0.62
Y_TOP=Y_BOTTOM+(SIDE_ROWS-1)*DX
BOTTOM_X0=X_LEFT-SIDE_LAYERS*DX
FLUID_X0=0.90
FLUID_Y0=0.72
VIEW_X0=0.25
VIEW_X1=3.65
VIEW_Y0=0.20
VIEW_Y1=4.25

pos=ti.Vector.field(2,dtype=ti.f32,shape=PARTICLE_N)
vel=ti.Vector.field(2,dtype=ti.f32,shape=PARTICLE_N)
pred_vel=ti.Vector.field(2,dtype=ti.f32,shape=PARTICLE_N)
acc=ti.Vector.field(2,dtype=ti.f32,shape=PARTICLE_N)
rho=ti.field(dtype=ti.f32,shape=PARTICLE_N)
pressure=ti.field(dtype=ti.f32,shape=PARTICLE_N)
tmp_pressure=ti.field(dtype=ti.f32,shape=PARTICLE_N)
pressure_diag=ti.field(dtype=ti.f32,shape=PARTICLE_N)
pressure_rhs=ti.field(dtype=ti.f32,shape=PARTICLE_N)
mass=ti.field(dtype=ti.f32,shape=PARTICLE_N)
edge=ti.field(dtype=ti.i32,shape=PARTICLE_N)
neighbour=ti.field(dtype=ti.i32,shape=PARTICLE_N*MAX_NEIGHBOUR)
neighbour_count=ti.field(dtype=ti.i32,shape=PARTICLE_N)
rest_density=ti.field(dtype=ti.f32,shape=())
avg_neighbour_count=ti.field(dtype=ti.f32,shape=())
max_pressure=ti.field(dtype=ti.f32,shape=())
max_divergence=ti.field(dtype=ti.f32,shape=())
max_speed=ti.field(dtype=ti.f32,shape=())
acc_limit_count=ti.field(dtype=ti.i32,shape=())
vel_limit_count=ti.field(dtype=ti.i32,shape=())
dt=ti.field(dtype=ti.f32,shape=())
pressure_scale=ti.field(dtype=ti.f32,shape=())

fluid_pos=ti.Vector.field(2,dtype=ti.f32,shape=FLUID_N)
fluid_color=ti.Vector.field(3,dtype=ti.f32,shape=FLUID_N)
visual_pressure=ti.field(dtype=ti.f32,shape=FLUID_N)
boundary_pos=ti.Vector.field(2,dtype=ti.f32,shape=BOUNDARY_N)
container_lines=ti.Vector.field(2,dtype=ti.f32,shape=8)

paused=False

@ti.func
def clamp_f(v:ti.f32,lo:ti.f32,hi:ti.f32)->ti.f32:
    return ti.min(ti.max(v,lo),hi)

@ti.func
def dist(a:vec2,b:vec2)->ti.f32:
    d=a-b
    return ti.sqrt(d.dot(d))

@ti.func
def W(r:ti.f32)->ti.f32:
    q=r/H
    res=0.0
    if q<=1.0:
        res=1.0-1.5*q*q+0.75*q*q*q
    elif q<=2.0:
        res=0.25*(2.0-q)*(2.0-q)*(2.0-q)
    return res*10.0/(7.0*PI*H*H)

@ti.func
def dW(r:ti.f32)->ti.f32:
    q=r/H
    res=0.0
    if q<=1.0:
        res=-3.0*q+2.25*q*q
    elif q<=2.0:
        res=-0.75*(2.0-q)*(2.0-q)
    return res*10.0/(7.0*PI*H*H*H)

@ti.func
def pressure_color(p:ti.f32)->vec3:
    t=clamp_f(p/(pressure_scale[None]+1e-5),0.0,1.0)
    cold=ti.Vector([0.20,0.50,1.00])
    hot=ti.Vector([1.00,0.34,0.16])
    return cold*(1.0-t)+hot*t

@ti.func
def to_canvas(p:vec2)->vec2:
    return ti.Vector([
        clamp_f((p[0]-VIEW_X0)/(VIEW_X1-VIEW_X0),0.0,1.0),
        clamp_f((p[1]-VIEW_Y0)/(VIEW_Y1-VIEW_Y0),0.0,1.0),
    ])

@ti.func
def rho0()->ti.f32:
    return ti.max(rest_density[None],1e-4)

@ti.func
def poisson_coeff(i:int,nb:int,r:ti.f32,grad:vec2)->ti.f32:
    rij=pos[i]-pos[nb]
    coeff=2.0*mass[nb]/rho0()*(-rij.dot(grad))/(r*r+ART_VISC_EPS*H*H)
    return ti.max(coeff,0.0)

@ti.kernel
def init_container_lines():
    p0=to_canvas(ti.Vector([X_LEFT,Y_BOTTOM]))
    p1=to_canvas(ti.Vector([X_RIGHT,Y_BOTTOM]))
    p2=to_canvas(ti.Vector([X_RIGHT,Y_TOP]))
    p3=to_canvas(ti.Vector([X_LEFT,Y_TOP]))
    container_lines[0]=p0
    container_lines[1]=p1
    container_lines[2]=p1
    container_lines[3]=p2
    container_lines[4]=p2
    container_lines[5]=p3
    container_lines[6]=p3
    container_lines[7]=p0

@ti.kernel
def init_particles():
    max_pressure[None]=1e-5
    max_divergence[None]=1e-5
    max_speed[None]=1e-5
    acc_limit_count[None]=0
    vel_limit_count[None]=0
    pressure_scale[None]=PRESSURE_COLOR_MIN
    for i,j in ti.ndrange(FLUID_SIDE,FLUID_SIDE):
        idx=i*FLUID_SIDE+j
        jitter_x=0.08*(ti.cast((idx*17+13)%29,ti.f32)/29.0-0.5)
        jitter_y=0.08*(ti.cast((idx*31+7)%37,ti.f32)/37.0-0.5)
        pos[idx]=ti.Vector([FLUID_X0+(ti.cast(i,ti.f32)+0.5+jitter_x)*DX,FLUID_Y0+(ti.cast(j,ti.f32)+0.5+jitter_y)*DX])
        vel[idx]=ti.Vector([0.0,0.0])
        pred_vel[idx]=ti.Vector([0.0,0.0])
        acc[idx]=ti.Vector([0.0,0.0])
        rho[idx]=1.0
        pressure[idx]=0.0
        tmp_pressure[idx]=0.0
        pressure_diag[idx]=1.0
        pressure_rhs[idx]=0.0
        mass[idx]=MASS
        edge[idx]=0
        neighbour_count[idx]=0
        visual_pressure[idx]=0.0
    base=FLUID_N
    for i,j in ti.ndrange(BOTTOM_COLS,BOTTOM_LAYERS):
        idx=base+i*BOTTOM_LAYERS+j
        pos[idx]=ti.Vector([BOTTOM_X0+ti.cast(i,ti.f32)*DX,Y_BOTTOM-ti.cast(j+1,ti.f32)*DX])
        vel[idx]=ti.Vector([0.0,0.0])
        pred_vel[idx]=ti.Vector([0.0,0.0])
        acc[idx]=ti.Vector([0.0,0.0])
        rho[idx]=1.0
        pressure[idx]=0.0
        tmp_pressure[idx]=0.0
        pressure_diag[idx]=1.0
        pressure_rhs[idx]=0.0
        mass[idx]=MASS
        edge[idx]=1
        neighbour_count[idx]=0
    base=FLUID_N+BOTTOM_N
    for i,j in ti.ndrange(SIDE_ROWS,SIDE_LAYERS):
        y=Y_BOTTOM+ti.cast(i,ti.f32)*DX
        left=base+i*SIDE_LAYERS+j
        right=base+SIDE_ROWS*SIDE_LAYERS+i*SIDE_LAYERS+j
        pos[left]=ti.Vector([X_LEFT-ti.cast(j+1,ti.f32)*DX,y])
        pos[right]=ti.Vector([X_RIGHT+ti.cast(j+1,ti.f32)*DX,y])
        vel[left]=ti.Vector([0.0,0.0])
        vel[right]=ti.Vector([0.0,0.0])
        pred_vel[left]=ti.Vector([0.0,0.0])
        pred_vel[right]=ti.Vector([0.0,0.0])
        acc[left]=ti.Vector([0.0,0.0])
        acc[right]=ti.Vector([0.0,0.0])
        rho[left]=1.0
        rho[right]=1.0
        pressure[left]=0.0
        pressure[right]=0.0
        tmp_pressure[left]=0.0
        tmp_pressure[right]=0.0
        pressure_diag[left]=1.0
        pressure_diag[right]=1.0
        pressure_rhs[left]=0.0
        pressure_rhs[right]=0.0
        mass[left]=MASS
        mass[right]=MASS
        edge[left]=1
        edge[right]=1
        neighbour_count[left]=0
        neighbour_count[right]=0

@ti.kernel
def find_neighbours():
    total=0
    for i in range(PARTICLE_N):
        count=0
        for j in range(PARTICLE_N):
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
def predict_velocity():
    acc_limit_count[None]=0
    for i in range(PARTICLE_N):
        if edge[i]==0:
            a=ti.Vector([0.0,G])
            for j in range(neighbour_count[i]):
                nb=neighbour[i*MAX_NEIGHBOUR+j]
                if edge[nb]==0:
                    rij=pos[i]-pos[nb]
                    r=ti.sqrt(rij.dot(rij))+1e-5
                    if r<2.0*H:
                        grad=rij/r*dW(r)
                        vij=vel[i]-vel[nb]
                        approach=vij.dot(rij)
                        if approach<0.0:
                            mu=H*approach/(r*r+ART_VISC_EPS*H*H)
                            pi_ij=-ART_VISC_ALPHA*SOUND_SPEED*mu/rho0()
                            a+=-mass[nb]*pi_ij*grad
                        a+=ETA*mass[nb]*(vel[nb]-vel[i])*W(r)/rho0()
            norm=a.norm()
            if norm>ACC_LIMIT:
                a=a/norm*ACC_LIMIT
                ti.atomic_add(acc_limit_count[None],1)
            acc[i]=a
            pred_vel[i]=vel[i]+dt[None]*a
        else:
            acc[i]=ti.Vector([0.0,0.0])
            pred_vel[i]=ti.Vector([0.0,0.0])
            pressure[i]=0.0

@ti.kernel
def build_pressure_system():
    max_divergence[None]=1e-5
    for i in range(PARTICLE_N):
        pressure_diag[i]=1.0
        pressure_rhs[i]=0.0
        if edge[i]==0:
            div=0.0
            diag=0.0
            for j in range(neighbour_count[i]):
                nb=neighbour[i*MAX_NEIGHBOUR+j]
                rij=pos[i]-pos[nb]
                r=ti.sqrt(rij.dot(rij))+1e-5
                if r<2.0*H:
                    grad=rij/r*dW(r)
                    div+=mass[nb]/rho0()*(pred_vel[nb]-pred_vel[i]).dot(grad)
                    coeff=poisson_coeff(i,nb,r,grad)
                    diag+=coeff
            pressure_diag[i]=ti.max(diag,POISSON_EPS)
            pressure_rhs[i]=-rho0()*div/dt[None]
            ti.atomic_max(max_divergence[None],ti.abs(div))
        else:
            pressure_rhs[i]=0.0

@ti.kernel
def solve_pressure_iter():
    for i in range(PARTICLE_N):
        if edge[i]==0:
            rhs=pressure_rhs[i]
            psum=0.0
            for j in range(neighbour_count[i]):
                nb=neighbour[i*MAX_NEIGHBOUR+j]
                if edge[nb]==0:
                    rij=pos[i]-pos[nb]
                    r=ti.sqrt(rij.dot(rij))+1e-5
                    if r<2.0*H:
                        grad=rij/r*dW(r)
                        coeff=poisson_coeff(i,nb,r,grad)
                        psum+=coeff*pressure[nb]
            new_pressure=ti.max(0.0,(rhs+psum)/pressure_diag[i])
            tmp_pressure[i]=PRESSURE_RELAX*new_pressure+(1.0-PRESSURE_RELAX)*pressure[i]
        else:
            tmp_pressure[i]=pressure[i]

@ti.kernel
def commit_pressure():
    for i in range(PARTICLE_N):
        pressure[i]=tmp_pressure[i]

@ti.kernel
def extrapolate_boundary_pressure():
    gravity=ti.Vector([0.0,G])
    for i in range(PARTICLE_N):
        if edge[i]==1:
            numerator=0.0
            denominator=0.0
            for j in range(neighbour_count[i]):
                nb=neighbour[i*MAX_NEIGHBOUR+j]
                if edge[nb]==0:
                    rij=pos[i]-pos[nb]
                    r=ti.sqrt(rij.dot(rij))+1e-5
                    if r<2.0*H:
                        w=W(r)
                        hydro=rho0()*gravity.dot(rij)
                        numerator+=(pressure[nb]+hydro)*w
                        denominator+=w
            if denominator>1e-6:
                target=ti.max(0.0,BOUNDARY_PRESSURE_SCALE*numerator/denominator)
                pressure[i]=(1.0-BOUNDARY_PRESSURE_RELAX)*pressure[i]+BOUNDARY_PRESSURE_RELAX*target
            else:
                pressure[i]=0.0

@ti.kernel
def apply_pressure():
    max_pressure[None]=1e-5
    max_speed[None]=1e-5
    vel_limit_count[None]=0
    for i in range(PARTICLE_N):
        if edge[i]==0:
            v=pred_vel[i]
            for j in range(neighbour_count[i]):
                nb=neighbour[i*MAX_NEIGHBOUR+j]
                rij=pos[i]-pos[nb]
                r=ti.sqrt(rij.dot(rij))+1e-5
                if r<2.0*H:
                    grad=rij/r*dW(r)
                    pressure_term=(pressure[i]+pressure[nb])/(rho0()*rho0())
                    if edge[nb]==1:
                        pressure_term=(pressure[i]+BOUNDARY_GRAD_SCALE*pressure[nb])/(rho0()*rho0())
                    v+=-dt[None]*mass[nb]*pressure_term*grad
            speed=v.norm()
            if speed>VEL_LIMIT:
                v=v/speed*VEL_LIMIT
                ti.atomic_add(vel_limit_count[None],1)
                speed=VEL_LIMIT
            ti.atomic_max(max_speed[None],speed)
            vel[i]=v
            ti.atomic_max(max_pressure[None],pressure[i])
        else:
            vel[i]=ti.Vector([0.0,0.0])

@ti.kernel
def integrate_position():
    for i in range(PARTICLE_N):
        if edge[i]==0:
            pos[i]+=dt[None]*vel[i]
            if pos[i][0]<X_LEFT:
                pos[i][0]=X_LEFT
                if vel[i][0]<0.0:
                    vel[i][0]=0.0
            if pos[i][0]<=X_LEFT+WALL_CONTACT_EPS:
                vel[i][1]*=WALL_TANGENT_DAMP
            if pos[i][0]>X_RIGHT:
                pos[i][0]=X_RIGHT
                if vel[i][0]>0.0:
                    vel[i][0]=0.0
            if pos[i][0]>=X_RIGHT-WALL_CONTACT_EPS:
                vel[i][1]*=WALL_TANGENT_DAMP
            if pos[i][1]<Y_BOTTOM:
                pos[i][1]=Y_BOTTOM
                if vel[i][1]<0.0:
                    vel[i][1]=0.0
            if pos[i][1]<=Y_BOTTOM+WALL_CONTACT_EPS:
                vel[i][0]*=WALL_TANGENT_DAMP
            if pos[i][1]>Y_TOP:
                pos[i][1]=Y_TOP
                if vel[i][1]>0.0:
                    vel[i][1]=0.0
            if pos[i][1]>=Y_TOP-WALL_CONTACT_EPS:
                vel[i][0]*=WALL_TANGENT_DAMP
        else:
            vel[i]=ti.Vector([0.0,0.0])

@ti.kernel
def update_pressure_scale():
    target=ti.max(max_pressure[None],PRESSURE_COLOR_MIN)
    pressure_scale[None]=PRESSURE_COLOR_SMOOTH*pressure_scale[None]+(1.0-PRESSURE_COLOR_SMOOTH)*target

@ti.kernel
def update_render_fields():
    for i in range(FLUID_N):
        fluid_pos[i]=to_canvas(pos[i])
        visual_pressure[i]=VISUAL_PRESSURE_SMOOTH*visual_pressure[i]+(1.0-VISUAL_PRESSURE_SMOOTH)*pressure[i]
        fluid_color[i]=pressure_color(visual_pressure[i])
    for i in range(BOUNDARY_N):
        boundary_pos[i]=to_canvas(pos[FLUID_N+i])

def substep():
    find_neighbours()
    compute_density()
    predict_velocity()
    build_pressure_system()
    solve_pressure()
    extrapolate_boundary_pressure()
    apply_pressure()
    integrate_position()

def solve_pressure():
    for _ in range(PRESSURE_ITERS):
        solve_pressure_iter()
        commit_pressure()

def init():
    dt[None]=DT_INIT
    init_container_lines()
    init_particles()
    find_neighbours()
    compute_density()
    init_rest_density()
    build_pressure_system()
    update_render_fields()

def handle_input(window):
    global paused
    for e in window.get_events(ti.ui.PRESS):
        if e.key==ti.ui.ESCAPE:
            window.running=False
        elif e.key==" ":
            paused=not paused
        elif e.key=="r" or e.key=="R":
            init()
        elif e.key=="[":
            dt[None]=max(0.00025,dt[None]*0.8)
        elif e.key=="]":
            dt[None]=min(0.006,dt[None]*1.25)

def render(canvas:ti.ui.Canvas):
    update_pressure_scale()
    update_render_fields()
    canvas.set_background_color((0.07,0.08,0.09))
    canvas.lines(container_lines,width=0.002,color=(0.55,0.58,0.62))
    canvas.circles(boundary_pos,radius=BOUNDARY_RADIUS,color=(0.40,0.42,0.44))
    canvas.circles(fluid_pos,radius=PART_RADIUS,per_vertex_color=fluid_color)

def render_gui(window):
    gui=window.get_gui()
    gui.begin("ISPH",0.02,0.02,0.27,0.25)
    gui.text(f"particles: {PARTICLE_N}")
    gui.text(f"fluid: {FLUID_N}")
    gui.text(f"rest rho: {rest_density[None]:.3f}")
    gui.text(f"avg nb: {avg_neighbour_count[None]:.1f}")
    gui.text(f"dt: {dt[None]:.4f}")
    gui.text(f"pressure iters: {PRESSURE_ITERS}")
    gui.text(f"color scale: {pressure_scale[None]:.2f}")
    gui.text(f"max div: {max_divergence[None]:.4f}")
    gui.text(f"max speed: {max_speed[None]:.3f}")
    gui.text(f"acc cap: {acc_limit_count[None]}")
    gui.text(f"vel cap: {vel_limit_count[None]}")
    gui.end()

def main():
    init()
    window=ti.ui.Window("ISPH 2D",(1000,1000))
    canvas=window.get_canvas()
    while window.running:
        handle_input(window)
        if not paused:
            for _ in range(SUBSTEPS):
                substep()
        render(canvas)
        render_gui(window)
        window.show()

if __name__=="__main__":
    main()
