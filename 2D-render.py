import taichi as ti

ti.init(arch=ti.cpu)

vec2=ti.types.vector(2,dtype=ti.f32)

GRID_SIZE=32
DOMAIN_MIN=0.1
DOMAIN_MAX=0.9
DOMAIN_LEN=DOMAIN_MAX-DOMAIN_MIN
GRID_LEN=DOMAIN_LEN/float(GRID_SIZE)
PARTICLE_COLS=30
PARTICLE_ROWS=30
NUM_PARTICLES=PARTICLE_COLS*PARTICLE_ROWS
FACE_X_NUM=(GRID_SIZE+1)*GRID_SIZE
FACE_Y_NUM=GRID_SIZE*(GRID_SIZE+1)
GRID_LINE_NUM=2*(GRID_SIZE+1)
GRID_LINE_VERTEX_NUM=GRID_LINE_NUM*2
PRESSURE_ITERS=60
SUBSTEPS=2
PART_RADIUS=0.004
RHO=1.0
GRAVITY=-1.0
BOUNCE=0.0
EPS=1e-6
WALL_MIN=DOMAIN_MIN+GRID_LEN
WALL_MAX=DOMAIN_MAX-GRID_LEN
WALL_TANGENT_DAMP=0.35
SURFACE_ALPHA=0.25
MIN_SOLVE_NEIGHBORS=2

TYPE_EMPTY=0
TYPE_SOLID=1
TYPE_LIQUID=2

dt=ti.field(dtype=ti.f32,shape=())
alpha=ti.field(dtype=ti.f32,shape=())

parts_pos=ti.Vector.field(2,dtype=ti.f32,shape=NUM_PARTICLES)
old_parts_vel=ti.Vector.field(2,dtype=ti.f32,shape=NUM_PARTICLES)
parts_vel=ti.Vector.field(2,dtype=ti.f32,shape=NUM_PARTICLES)
new_part_vel=ti.Vector.field(2,dtype=ti.f32,shape=NUM_PARTICLES)
parts_grid=ti.field(dtype=ti.i32,shape=NUM_PARTICLES)

grid_type=ti.field(dtype=ti.i32,shape=GRID_SIZE*GRID_SIZE)
grid_cnt=ti.field(dtype=ti.i32,shape=GRID_SIZE*GRID_SIZE)
grid_solve=ti.field(dtype=ti.i32,shape=GRID_SIZE*GRID_SIZE)
grid_pres=ti.field(dtype=ti.f32,shape=GRID_SIZE*GRID_SIZE)

grid_velx=ti.field(dtype=ti.f32,shape=FACE_X_NUM)
grid_vely=ti.field(dtype=ti.f32,shape=FACE_Y_NUM)
old_grid_vel_x=ti.field(dtype=ti.f32,shape=FACE_X_NUM)
old_grid_vel_y=ti.field(dtype=ti.f32,shape=FACE_Y_NUM)
grid_weightx=ti.field(dtype=ti.f32,shape=FACE_X_NUM)
grid_weighty=ti.field(dtype=ti.f32,shape=FACE_Y_NUM)
grid_validx=ti.field(dtype=ti.i32,shape=FACE_X_NUM)
grid_validy=ti.field(dtype=ti.i32,shape=FACE_Y_NUM)
grid_lines=ti.Vector.field(2,dtype=ti.f32,shape=GRID_LINE_VERTEX_NUM)

paused=False

@ti.func
def clamp_i(v:int,lo:int,hi:int)->int:
    return ti.min(ti.max(v,lo),hi)

@ti.func
def cell_id(i:int,j:int)->int:
    return i*GRID_SIZE+j

@ti.func
def face_x_id(i:int,j:int)->int:
    return i*GRID_SIZE+j

@ti.func
def face_y_id(i:int,j:int)->int:
    return i*(GRID_SIZE+1)+j

@ti.func
def cell_pos(i:int,j:int)->vec2:
    return ti.Vector([DOMAIN_MIN+(ti.cast(i,ti.f32)+0.5)*GRID_LEN,DOMAIN_MIN+(ti.cast(j,ti.f32)+0.5)*GRID_LEN])

@ti.func
def face_x_pos(i:int,j:int)->vec2:
    return ti.Vector([DOMAIN_MIN+ti.cast(i,ti.f32)*GRID_LEN,DOMAIN_MIN+(ti.cast(j,ti.f32)+0.5)*GRID_LEN])

@ti.func
def face_y_pos(i:int,j:int)->vec2:
    return ti.Vector([DOMAIN_MIN+(ti.cast(i,ti.f32)+0.5)*GRID_LEN,DOMAIN_MIN+ti.cast(j,ti.f32)*GRID_LEN])

@ti.func
def kern(r:ti.f32)->ti.f32:
    a=ti.abs(r)
    ans=0.0
    if a<0.5:
        ans=0.75-a*a
    elif a<1.5:
        ans=0.5*(1.5-a)*(1.5-a)
    return ans

@ti.func
def weight(dr:vec2)->ti.f32:
    r=dr/GRID_LEN
    return kern(r[0])*kern(r[1])

@ti.func
def is_solid_cell(i:int,j:int)->int:
    ans=0
    if i<0:
        ans=1
    elif i>=GRID_SIZE:
        ans=1
    elif j<0:
        ans=1
    elif j>=GRID_SIZE:
        ans=1
    elif grid_type[cell_id(i,j)]==TYPE_SOLID:
        ans=1
    return ans

@ti.func
def is_liquid_cell(i:int,j:int)->int:
    ans=0
    if i>=0 and i<GRID_SIZE and j>=0 and j<GRID_SIZE:
        if grid_type[cell_id(i,j)]==TYPE_LIQUID:
            ans=1
    return ans

@ti.func
def liquid_neighbor_count(i:int,j:int)->int:
    cnt=0
    cnt+=is_liquid_cell(i-1,j)
    cnt+=is_liquid_cell(i+1,j)
    cnt+=is_liquid_cell(i,j-1)
    cnt+=is_liquid_cell(i,j+1)
    return cnt

@ti.func
def is_pressure_cell(i:int,j:int)->int:
    ans=0
    if i>=0 and i<GRID_SIZE and j>=0 and j<GRID_SIZE:
        if grid_solve[cell_id(i,j)]==1:
            ans=1
    return ans

@ti.func
def pressure_at(i:int,j:int)->ti.f32:
    ans=0.0
    if i>=0 and i<GRID_SIZE and j>=0 and j<GRID_SIZE:
        if grid_solve[cell_id(i,j)]==1:
            ans=grid_pres[cell_id(i,j)]
    return ans

@ti.func
def has_liquid_x(fi:int,fj:int)->int:
    ans=0
    if fi>0:
        if is_liquid_cell(fi-1,fj)==1:
            ans=1
    if fi<GRID_SIZE:
        if is_liquid_cell(fi,fj)==1:
            ans=1
    return ans

@ti.func
def has_liquid_y(fi:int,fj:int)->int:
    ans=0
    if fj>0:
        if is_liquid_cell(fi,fj-1)==1:
            ans=1
    if fj<GRID_SIZE:
        if is_liquid_cell(fi,fj)==1:
            ans=1
    return ans

@ti.func
def part_cell(pos:vec2)->int:
    ci=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN),ti.i32)
    cj=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN),ti.i32)
    ci=clamp_i(ci,1,GRID_SIZE-2)
    cj=clamp_i(cj,1,GRID_SIZE-2)
    return cell_id(ci,cj)

@ti.func
def sample_x(pos:vec2,vel:ti.template())->ti.f32:
    ans=0.0
    wsum=0.0
    base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN),ti.i32)
    base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
    for di in ti.static(range(-1,3)):
        for dj in ti.static(range(-1,3)):
            fi=base_i+di
            fj=base_j+dj
            if fi>=0 and fi<=GRID_SIZE and fj>=0 and fj<GRID_SIZE:
                fid=face_x_id(fi,fj)
                if grid_validx[fid]==1:
                    w=weight(face_x_pos(fi,fj)-pos)
                    ans+=vel[fid]*w
                    wsum+=w
    if wsum>EPS:
        ans/=wsum
    return ans

@ti.func
def sample_y(pos:vec2,vel:ti.template())->ti.f32:
    ans=0.0
    wsum=0.0
    base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
    base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN),ti.i32)
    for di in ti.static(range(-1,3)):
        for dj in ti.static(range(-1,3)):
            fi=base_i+di
            fj=base_j+dj
            if fi>=0 and fi<GRID_SIZE and fj>=0 and fj<=GRID_SIZE:
                fid=face_y_id(fi,fj)
                if grid_validy[fid]==1:
                    w=weight(face_y_pos(fi,fj)-pos)
                    ans+=vel[fid]*w
                    wsum+=w
    if wsum>EPS:
        ans/=wsum
    return ans

@ti.func
def sample_vel(pos:vec2,velx:ti.template(),vely:ti.template())->vec2:
    return ti.Vector([sample_x(pos,velx),sample_y(pos,vely)])

@ti.kernel
def init_grid_lines():
    for i in range(GRID_SIZE+1):
        x=DOMAIN_MIN+ti.cast(i,ti.f32)*GRID_LEN
        y=DOMAIN_MIN+ti.cast(i,ti.f32)*GRID_LEN
        grid_lines[i*2]=ti.Vector([x,DOMAIN_MIN])
        grid_lines[i*2+1]=ti.Vector([x,DOMAIN_MAX])
        base=(GRID_SIZE+1)*2+i*2
        grid_lines[base]=ti.Vector([DOMAIN_MIN,y])
        grid_lines[base+1]=ti.Vector([DOMAIN_MAX,y])

@ti.kernel
def init_particles():
    for i in range(NUM_PARTICLES):
        col=i%PARTICLE_COLS
        row=i//PARTICLE_COLS
        jitter_x=ti.cast((i*17+13)%29,ti.f32)/29.0
        jitter_y=ti.cast((i*31+7)%37,ti.f32)/37.0
        x=0.22+(ti.cast(col,ti.f32)+0.2+0.45*jitter_x)*GRID_LEN/1.7
        y=0.18+(ti.cast(row,ti.f32)+0.2+0.45*jitter_y)*GRID_LEN/1.7
        parts_pos[i]=ti.Vector([x,y])
        parts_vel[i]=ti.Vector([0.0,0.0])
        old_parts_vel[i]=ti.Vector([0.0,0.0])
        new_part_vel[i]=ti.Vector([0.0,0.0])
        parts_grid[i]=part_cell(parts_pos[i])
    for i in range(FACE_X_NUM):
        grid_velx[i]=0.0
        old_grid_vel_x[i]=0.0
        grid_weightx[i]=0.0
        grid_validx[i]=0
    for i in range(FACE_Y_NUM):
        grid_vely[i]=0.0
        old_grid_vel_y[i]=0.0
        grid_weighty[i]=0.0
        grid_validy[i]=0
    for i in range(GRID_SIZE*GRID_SIZE):
        grid_type[i]=TYPE_EMPTY
        grid_cnt[i]=0
        grid_solve[i]=0
        grid_pres[i]=0.0

@ti.kernel
def P2G():
    for i in range(GRID_SIZE*GRID_SIZE):
        gi=i//GRID_SIZE
        gj=i%GRID_SIZE
        grid_cnt[i]=0
        grid_solve[i]=0
        grid_pres[i]=0.0
        if gi==0 or gi==GRID_SIZE-1 or gj==0 or gj==GRID_SIZE-1:
            grid_type[i]=TYPE_SOLID
        else:
            grid_type[i]=TYPE_EMPTY
    for i in range(FACE_X_NUM):
        grid_velx[i]=0.0
        grid_weightx[i]=0.0
        grid_validx[i]=0
    for i in range(FACE_Y_NUM):
        grid_vely[i]=0.0
        grid_weighty[i]=0.0
        grid_validy[i]=0
    for p in range(NUM_PARTICLES):
        old_parts_vel[p]=parts_vel[p]
        gid=part_cell(parts_pos[p])
        parts_grid[p]=gid
        if grid_type[gid]!=TYPE_SOLID:
            grid_type[gid]=TYPE_LIQUID
            ti.atomic_add(grid_cnt[gid],1)
    for i,j in ti.ndrange(GRID_SIZE,GRID_SIZE):
        gid=cell_id(i,j)
        if grid_type[gid]==TYPE_LIQUID:
            if liquid_neighbor_count(i,j)>=MIN_SOLVE_NEIGHBORS:
                grid_solve[gid]=1
    for p in range(NUM_PARTICLES):
        pos=parts_pos[p]
        base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN),ti.i32)
        base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        for di in ti.static(range(-1,3)):
            for dj in ti.static(range(-1,3)):
                fi=base_i+di
                fj=base_j+dj
                if fi>=0 and fi<=GRID_SIZE and fj>=0 and fj<GRID_SIZE:
                    w=weight(face_x_pos(fi,fj)-pos)
                    fid=face_x_id(fi,fj)
                    ti.atomic_add(grid_velx[fid],parts_vel[p][0]*w)
                    ti.atomic_add(grid_weightx[fid],w)
        base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN),ti.i32)
        for di in ti.static(range(-1,3)):
            for dj in ti.static(range(-1,3)):
                fi=base_i+di
                fj=base_j+dj
                if fi>=0 and fi<GRID_SIZE and fj>=0 and fj<=GRID_SIZE:
                    w=weight(face_y_pos(fi,fj)-pos)
                    fid=face_y_id(fi,fj)
                    ti.atomic_add(grid_vely[fid],parts_vel[p][1]*w)
                    ti.atomic_add(grid_weighty[fid],w)
    for i in range(FACE_X_NUM):
        fi=i//GRID_SIZE
        fj=i%GRID_SIZE
        if has_liquid_x(fi,fj)==1 and fi>0 and fi<GRID_SIZE and is_solid_cell(fi-1,fj)==0 and is_solid_cell(fi,fj)==0:
            grid_validx[i]=1
        if grid_weightx[i]>EPS:
            grid_velx[i]/=grid_weightx[i]
        if fi==0 or fi==GRID_SIZE:
            grid_velx[i]=0.0
        elif is_solid_cell(fi-1,fj)==1 or is_solid_cell(fi,fj)==1:
            grid_velx[i]=0.0
        old_grid_vel_x[i]=grid_velx[i]
    for i in range(FACE_Y_NUM):
        fi=i//(GRID_SIZE+1)
        fj=i%(GRID_SIZE+1)
        if has_liquid_y(fi,fj)==1 and fj>0 and fj<GRID_SIZE and is_solid_cell(fi,fj-1)==0 and is_solid_cell(fi,fj)==0:
            grid_validy[i]=1
        if grid_weighty[i]>EPS:
            grid_vely[i]/=grid_weighty[i]
        if fj==0 or fj==GRID_SIZE:
            grid_vely[i]=0.0
        elif is_solid_cell(fi,fj-1)==1 or is_solid_cell(fi,fj)==1:
            grid_vely[i]=0.0
        old_grid_vel_y[i]=grid_vely[i]

@ti.kernel
def add_force():
    for i in range(FACE_Y_NUM):
        fi=i//(GRID_SIZE+1)
        fj=i%(GRID_SIZE+1)
        if fj>0 and fj<GRID_SIZE:
            if is_solid_cell(fi,fj-1)==0 and is_solid_cell(fi,fj)==0:
                if is_liquid_cell(fi,fj-1)==1 or is_liquid_cell(fi,fj)==1:
                    grid_vely[i]+=GRAVITY*dt[None]

@ti.kernel
def solve_pressure():
    coef=RHO*GRID_LEN*GRID_LEN/dt[None]
    for _ in range(PRESSURE_ITERS):
        for i,j in ti.ndrange(GRID_SIZE,GRID_SIZE):
            gid=cell_id(i,j)
            if grid_solve[gid]==1:
                div=(grid_velx[face_x_id(i+1,j)]-grid_velx[face_x_id(i,j)]+grid_vely[face_y_id(i,j+1)]-grid_vely[face_y_id(i,j)])/GRID_LEN
                cnt=0.0
                psum=0.0
                if is_solid_cell(i-1,j)==0:
                    cnt+=1.0
                    psum+=pressure_at(i-1,j)
                if is_solid_cell(i+1,j)==0:
                    cnt+=1.0
                    psum+=pressure_at(i+1,j)
                if is_solid_cell(i,j-1)==0:
                    cnt+=1.0
                    psum+=pressure_at(i,j-1)
                if is_solid_cell(i,j+1)==0:
                    cnt+=1.0
                    psum+=pressure_at(i,j+1)
                if cnt>0.0:
                    grid_pres[gid]=(psum-coef*div)/cnt

@ti.kernel
def project_velocity():
    scale=dt[None]/(RHO*GRID_LEN)
    for i in range(FACE_X_NUM):
        fi=i//GRID_SIZE
        fj=i%GRID_SIZE
        if fi==0 or fi==GRID_SIZE:
            grid_velx[i]=0.0
        elif is_solid_cell(fi-1,fj)==1 or is_solid_cell(fi,fj)==1:
            grid_velx[i]=0.0
        elif is_pressure_cell(fi-1,fj)==1 or is_pressure_cell(fi,fj)==1:
            grid_velx[i]-=scale*(pressure_at(fi,fj)-pressure_at(fi-1,fj))
        elif is_liquid_cell(fi-1,fj)==1 or is_liquid_cell(fi,fj)==1:
            grid_velx[i]=grid_velx[i]
        else:
            grid_velx[i]=0.0
    for i in range(FACE_Y_NUM):
        fi=i//(GRID_SIZE+1)
        fj=i%(GRID_SIZE+1)
        if fj==0 or fj==GRID_SIZE:
            grid_vely[i]=0.0
        elif is_solid_cell(fi,fj-1)==1 or is_solid_cell(fi,fj)==1:
            grid_vely[i]=0.0
        elif is_pressure_cell(fi,fj-1)==1 or is_pressure_cell(fi,fj)==1:
            grid_vely[i]-=scale*(pressure_at(fi,fj)-pressure_at(fi,fj-1))
        elif is_liquid_cell(fi,fj-1)==1 or is_liquid_cell(fi,fj)==1:
            grid_vely[i]=grid_vely[i]
        else:
            grid_vely[i]=0.0

@ti.kernel
def G2P():
    for i in range(NUM_PARTICLES):
        pos=parts_pos[i]
        gid=parts_grid[i]
        pic=sample_vel(pos,grid_velx,grid_vely)
        old_grid=sample_vel(pos,old_grid_vel_x,old_grid_vel_y)
        flip=old_parts_vel[i]+pic-old_grid
        ratio=alpha[None]
        if grid_solve[gid]==0:
            ratio=ti.min(ratio,SURFACE_ALPHA)
        if pos[0]<WALL_MIN+GRID_LEN or pos[0]>WALL_MAX-GRID_LEN:
            ratio=ti.min(ratio,SURFACE_ALPHA)
        if pos[1]<WALL_MIN+GRID_LEN or pos[1]>WALL_MAX-GRID_LEN:
            ratio=ti.min(ratio,SURFACE_ALPHA)
        new_part_vel[i]=(1.0-ratio)*pic+ratio*flip

@ti.kernel
def update_pos():
    lo=WALL_MIN+PART_RADIUS
    hi=WALL_MAX-PART_RADIUS
    for i in range(NUM_PARTICLES):
        vel=new_part_vel[i]
        speed=vel.norm()
        if speed>3.0:
            vel=vel/speed*3.0
        pos=parts_pos[i]+vel*dt[None]
        if pos[0]<lo:
            pos[0]=lo
            if vel[0]<0.0:
                vel[0]=-vel[0]*BOUNCE
            vel[1]*=WALL_TANGENT_DAMP
        if pos[0]>hi:
            pos[0]=hi
            if vel[0]>0.0:
                vel[0]=-vel[0]*BOUNCE
            vel[1]*=WALL_TANGENT_DAMP
        if pos[1]<lo:
            pos[1]=lo
            if vel[1]<0.0:
                vel[1]=-vel[1]*BOUNCE
            vel[0]*=WALL_TANGENT_DAMP
        if pos[1]>hi:
            pos[1]=hi
            if vel[1]>0.0:
                vel[1]=-vel[1]*BOUNCE
            vel[0]*=WALL_TANGENT_DAMP
        parts_pos[i]=pos
        parts_vel[i]=vel

def substep():
    P2G()
    add_force()
    solve_pressure()
    project_velocity()
    G2P()
    update_pos()

def init():
    dt[None]=0.005
    alpha[None]=0.95
    init_grid_lines()
    init_particles()

def clamp(v,lo,hi):
    return max(lo,min(v,hi))

def handle_input(window):
    global paused
    for e in window.get_events(ti.ui.PRESS):
        if e.key==ti.ui.ESCAPE:
            window.running=False
        elif e.key==" ":
            paused=not paused
        elif e.key=="r" or e.key=="R":
            init()
    if window.is_pressed(ti.ui.LEFT):
        alpha[None]=clamp(alpha[None]-0.01,0.0,1.0)
    if window.is_pressed(ti.ui.RIGHT):
        alpha[None]=clamp(alpha[None]+0.01,0.0,1.0)
    if window.is_pressed(ti.ui.DOWN):
        dt[None]=clamp(dt[None]*0.98,0.001,0.02)
    if window.is_pressed(ti.ui.UP):
        dt[None]=clamp(dt[None]*1.02,0.001,0.02)

def render(canvas:ti.ui.Canvas):
    canvas.set_background_color((0.07,0.08,0.09))
    canvas.lines(grid_lines,width=0.001,color=(0.24,0.27,0.30))
    canvas.circles(parts_pos,PART_RADIUS,color=(0.22,0.55,0.95))

def main():
    init()
    window=ti.ui.Window("2D FLIP Fluid",(1000,1000))
    canvas=window.get_canvas()
    while window.running:
        handle_input(window)
        if not paused:
            for _ in range(SUBSTEPS):
                substep()
        render(canvas)
        window.show()

if __name__=="__main__":
    main()
