import taichi as ti

ti.init(arch=ti.cpu)

vec3=ti.types.vector(3,dtype=ti.f32)

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
NUM_PARTICLES=LIQUID_CELL_X*LIQUID_CELL_Y*LIQUID_CELL_Z*PARTICLES_PER_CELL
CELL_NUM=GRID_SIZE*GRID_SIZE*GRID_SIZE
FACE_X_NUM=(GRID_SIZE+1)*GRID_SIZE*GRID_SIZE
FACE_Y_NUM=GRID_SIZE*(GRID_SIZE+1)*GRID_SIZE
FACE_Z_NUM=GRID_SIZE*GRID_SIZE*(GRID_SIZE+1)
BOX_LINE_VERTEX_NUM=24
GRID_LINE_VERTEX_NUM=6*(GRID_SIZE+1)*(GRID_SIZE+1)
PRESSURE_ITERS=70
SUBSTEPS=2
DT_INIT=0.004
DT_MIN=0.001
DT_MAX=0.010
PART_RADIUS=0.0032
RHO=1.0
GRAVITY=-1.0
ETA=0.0008
BOUNCE=0.0
EPS=1e-6
WALL_MIN=DOMAIN_MIN+GRID_LEN
WALL_MAX=DOMAIN_MAX-GRID_LEN
WALL_TANGENT_DAMP=0.92
SURFACE_ALPHA=0.95
EXTRAPOLATE_ITERS=1
MIN_SOLVE_NEIGHBORS=1
IDP_BETA=0.5
IDP_MAX_ERROR=0.30
DENSITY_REST_THRESHOLD=1.0
DENSITY_LIQUID_RATIO=0.10
MAX_PARTICLE_SPEED=5.0

TYPE_EMPTY=0
TYPE_SOLID=1
TYPE_LIQUID=2

dt=ti.field(dtype=ti.f32,shape=())
alpha=ti.field(dtype=ti.f32,shape=())
pressure_abs_max=ti.field(dtype=ti.f32,shape=())
rest_density=ti.field(dtype=ti.f32,shape=())
density_sum=ti.field(dtype=ti.f32,shape=())
density_cnt=ti.field(dtype=ti.f32,shape=())
max_speed=ti.field(dtype=ti.f32,shape=())
vel_limit_count=ti.field(dtype=ti.i32,shape=())

parts_pos=ti.Vector.field(3,dtype=ti.f32,shape=NUM_PARTICLES)
old_parts_vel=ti.Vector.field(3,dtype=ti.f32,shape=NUM_PARTICLES)
parts_vel=ti.Vector.field(3,dtype=ti.f32,shape=NUM_PARTICLES)
new_part_vel=ti.Vector.field(3,dtype=ti.f32,shape=NUM_PARTICLES)
parts_grid=ti.field(dtype=ti.i32,shape=NUM_PARTICLES)

grid_type=ti.field(dtype=ti.i32,shape=CELL_NUM)
grid_cnt=ti.field(dtype=ti.i32,shape=CELL_NUM)
grid_solve=ti.field(dtype=ti.i32,shape=CELL_NUM)
grid_pres=ti.field(dtype=ti.f32,shape=CELL_NUM)
grid_density=ti.field(dtype=ti.f32,shape=CELL_NUM)

grid_velx=ti.field(dtype=ti.f32,shape=FACE_X_NUM)
grid_vely=ti.field(dtype=ti.f32,shape=FACE_Y_NUM)
grid_velz=ti.field(dtype=ti.f32,shape=FACE_Z_NUM)
tmp_grid_velx=ti.field(dtype=ti.f32,shape=FACE_X_NUM)
tmp_grid_vely=ti.field(dtype=ti.f32,shape=FACE_Y_NUM)
tmp_grid_velz=ti.field(dtype=ti.f32,shape=FACE_Z_NUM)
old_grid_vel_x=ti.field(dtype=ti.f32,shape=FACE_X_NUM)
old_grid_vel_y=ti.field(dtype=ti.f32,shape=FACE_Y_NUM)
old_grid_vel_z=ti.field(dtype=ti.f32,shape=FACE_Z_NUM)
grid_weightx=ti.field(dtype=ti.f32,shape=FACE_X_NUM)
grid_weighty=ti.field(dtype=ti.f32,shape=FACE_Y_NUM)
grid_weightz=ti.field(dtype=ti.f32,shape=FACE_Z_NUM)
grid_validx=ti.field(dtype=ti.i32,shape=FACE_X_NUM)
grid_validy=ti.field(dtype=ti.i32,shape=FACE_Y_NUM)
grid_validz=ti.field(dtype=ti.i32,shape=FACE_Z_NUM)
tmp_validx=ti.field(dtype=ti.i32,shape=FACE_X_NUM)
tmp_validy=ti.field(dtype=ti.i32,shape=FACE_Y_NUM)
tmp_validz=ti.field(dtype=ti.i32,shape=FACE_Z_NUM)

box_lines=ti.Vector.field(3,dtype=ti.f32,shape=BOX_LINE_VERTEX_NUM)
grid_lines=ti.Vector.field(3,dtype=ti.f32,shape=GRID_LINE_VERTEX_NUM)

paused=False

@ti.func
def clamp_i(v:int,lo:int,hi:int)->int:
    return ti.min(ti.max(v,lo),hi)

@ti.func
def cell_id(i:int,j:int,k:int)->int:
    return (i*GRID_SIZE+j)*GRID_SIZE+k

@ti.func
def face_x_id(i:int,j:int,k:int)->int:
    return (i*GRID_SIZE+j)*GRID_SIZE+k

@ti.func
def face_y_id(i:int,j:int,k:int)->int:
    return (i*(GRID_SIZE+1)+j)*GRID_SIZE+k

@ti.func
def face_z_id(i:int,j:int,k:int)->int:
    return (i*GRID_SIZE+j)*(GRID_SIZE+1)+k

@ti.func
def cell_pos(i:int,j:int,k:int)->vec3:
    return ti.Vector([
        DOMAIN_MIN+(ti.cast(i,ti.f32)+0.5)*GRID_LEN,
        DOMAIN_MIN+(ti.cast(j,ti.f32)+0.5)*GRID_LEN,
        DOMAIN_MIN+(ti.cast(k,ti.f32)+0.5)*GRID_LEN,
    ])

@ti.func
def face_x_pos(i:int,j:int,k:int)->vec3:
    return ti.Vector([
        DOMAIN_MIN+ti.cast(i,ti.f32)*GRID_LEN,
        DOMAIN_MIN+(ti.cast(j,ti.f32)+0.5)*GRID_LEN,
        DOMAIN_MIN+(ti.cast(k,ti.f32)+0.5)*GRID_LEN,
    ])

@ti.func
def face_y_pos(i:int,j:int,k:int)->vec3:
    return ti.Vector([
        DOMAIN_MIN+(ti.cast(i,ti.f32)+0.5)*GRID_LEN,
        DOMAIN_MIN+ti.cast(j,ti.f32)*GRID_LEN,
        DOMAIN_MIN+(ti.cast(k,ti.f32)+0.5)*GRID_LEN,
    ])

@ti.func
def face_z_pos(i:int,j:int,k:int)->vec3:
    return ti.Vector([
        DOMAIN_MIN+(ti.cast(i,ti.f32)+0.5)*GRID_LEN,
        DOMAIN_MIN+(ti.cast(j,ti.f32)+0.5)*GRID_LEN,
        DOMAIN_MIN+ti.cast(k,ti.f32)*GRID_LEN,
    ])

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
def weight(dr:vec3)->ti.f32:
    r=dr/GRID_LEN
    return kern(r[0])*kern(r[1])*kern(r[2])

@ti.func
def is_solid_cell(i:int,j:int,k:int)->int:
    ans=0
    if i<0:
        ans=1
    elif i>=GRID_SIZE:
        ans=1
    elif j<0:
        ans=1
    elif j>=GRID_SIZE:
        ans=1
    elif k<0:
        ans=1
    elif k>=GRID_SIZE:
        ans=1
    elif grid_type[cell_id(i,j,k)]==TYPE_SOLID:
        ans=1
    return ans

@ti.func
def is_liquid_cell(i:int,j:int,k:int)->int:
    ans=0
    if i>=0 and i<GRID_SIZE and j>=0 and j<GRID_SIZE and k>=0 and k<GRID_SIZE:
        if grid_type[cell_id(i,j,k)]==TYPE_LIQUID:
            ans=1
    return ans

@ti.func
def liquid_neighbor_count(i:int,j:int,k:int)->int:
    cnt=0
    cnt+=is_liquid_cell(i-1,j,k)
    cnt+=is_liquid_cell(i+1,j,k)
    cnt+=is_liquid_cell(i,j-1,k)
    cnt+=is_liquid_cell(i,j+1,k)
    cnt+=is_liquid_cell(i,j,k-1)
    cnt+=is_liquid_cell(i,j,k+1)
    return cnt

@ti.func
def is_pressure_cell(i:int,j:int,k:int)->int:
    ans=0
    if i>=0 and i<GRID_SIZE and j>=0 and j<GRID_SIZE and k>=0 and k<GRID_SIZE:
        if grid_solve[cell_id(i,j,k)]==1:
            ans=1
    return ans

@ti.func
def pressure_at(i:int,j:int,k:int)->ti.f32:
    ans=0.0
    if i>=0 and i<GRID_SIZE and j>=0 and j<GRID_SIZE and k>=0 and k<GRID_SIZE:
        if grid_solve[cell_id(i,j,k)]==1:
            ans=grid_pres[cell_id(i,j,k)]
    return ans

@ti.func
def target_divergence(gid:int)->ti.f32:
    ans=0.0
    if rest_density[None]>EPS:
        err=(grid_density[gid]-rest_density[None])/rest_density[None]
        if err>0.0:
            err=ti.min(err,IDP_MAX_ERROR)
            ans=IDP_BETA*err/dt[None]
    return ans

@ti.func
def solid_face_x(fi:int,fj:int,fk:int)->int:
    ans=0
    if fi<=0 or fi>=GRID_SIZE:
        ans=1
    elif fj<0 or fj>=GRID_SIZE or fk<0 or fk>=GRID_SIZE:
        ans=1
    elif is_solid_cell(fi-1,fj,fk)==1 or is_solid_cell(fi,fj,fk)==1:
        ans=1
    return ans

@ti.func
def solid_face_y(fi:int,fj:int,fk:int)->int:
    ans=0
    if fj<=0 or fj>=GRID_SIZE:
        ans=1
    elif fi<0 or fi>=GRID_SIZE or fk<0 or fk>=GRID_SIZE:
        ans=1
    elif is_solid_cell(fi,fj-1,fk)==1 or is_solid_cell(fi,fj,fk)==1:
        ans=1
    return ans

@ti.func
def solid_face_z(fi:int,fj:int,fk:int)->int:
    ans=0
    if fk<=0 or fk>=GRID_SIZE:
        ans=1
    elif fi<0 or fi>=GRID_SIZE or fj<0 or fj>=GRID_SIZE:
        ans=1
    elif is_solid_cell(fi,fj,fk-1)==1 or is_solid_cell(fi,fj,fk)==1:
        ans=1
    return ans

@ti.func
def has_liquid_x(fi:int,fj:int,fk:int)->int:
    ans=0
    if fi>0 and is_liquid_cell(fi-1,fj,fk)==1:
        ans=1
    if fi<GRID_SIZE and is_liquid_cell(fi,fj,fk)==1:
        ans=1
    return ans

@ti.func
def has_liquid_y(fi:int,fj:int,fk:int)->int:
    ans=0
    if fj>0 and is_liquid_cell(fi,fj-1,fk)==1:
        ans=1
    if fj<GRID_SIZE and is_liquid_cell(fi,fj,fk)==1:
        ans=1
    return ans

@ti.func
def has_liquid_z(fi:int,fj:int,fk:int)->int:
    ans=0
    if fk>0 and is_liquid_cell(fi,fj,fk-1)==1:
        ans=1
    if fk<GRID_SIZE and is_liquid_cell(fi,fj,fk)==1:
        ans=1
    return ans

@ti.func
def part_cell(pos:vec3)->int:
    ci=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN),ti.i32)
    cj=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN),ti.i32)
    ck=ti.cast(ti.floor((pos[2]-DOMAIN_MIN)/GRID_LEN),ti.i32)
    ci=clamp_i(ci,1,GRID_SIZE-2)
    cj=clamp_i(cj,1,GRID_SIZE-2)
    ck=clamp_i(ck,1,GRID_SIZE-2)
    return cell_id(ci,cj,ck)

@ti.func
def lap_x_value(fi:int,fj:int,fk:int,center:ti.f32)->ti.f32:
    ans=center
    if fi>=0 and fi<=GRID_SIZE and fj>=0 and fj<GRID_SIZE and fk>=0 and fk<GRID_SIZE:
        fid=face_x_id(fi,fj,fk)
        if grid_validx[fid]==1:
            ans=grid_velx[fid]
        elif solid_face_x(fi,fj,fk)==1:
            ans=0.0
    return ans

@ti.func
def lap_y_value(fi:int,fj:int,fk:int,center:ti.f32)->ti.f32:
    ans=center
    if fi>=0 and fi<GRID_SIZE and fj>=0 and fj<=GRID_SIZE and fk>=0 and fk<GRID_SIZE:
        fid=face_y_id(fi,fj,fk)
        if grid_validy[fid]==1:
            ans=grid_vely[fid]
        elif solid_face_y(fi,fj,fk)==1:
            ans=0.0
    return ans

@ti.func
def lap_z_value(fi:int,fj:int,fk:int,center:ti.f32)->ti.f32:
    ans=center
    if fi>=0 and fi<GRID_SIZE and fj>=0 and fj<GRID_SIZE and fk>=0 and fk<=GRID_SIZE:
        fid=face_z_id(fi,fj,fk)
        if grid_validz[fid]==1:
            ans=grid_velz[fid]
        elif solid_face_z(fi,fj,fk)==1:
            ans=0.0
    return ans

@ti.func
def laplacian_x(fi:int,fj:int,fk:int)->ti.f32:
    center=grid_velx[face_x_id(fi,fj,fk)]
    left=lap_x_value(fi-1,fj,fk,center)
    right=lap_x_value(fi+1,fj,fk,center)
    down=lap_x_value(fi,fj-1,fk,center)
    up=lap_x_value(fi,fj+1,fk,center)
    back=lap_x_value(fi,fj,fk-1,center)
    front=lap_x_value(fi,fj,fk+1,center)
    return (left+right+down+up+back+front-6.0*center)/(GRID_LEN*GRID_LEN)

@ti.func
def laplacian_y(fi:int,fj:int,fk:int)->ti.f32:
    center=grid_vely[face_y_id(fi,fj,fk)]
    left=lap_y_value(fi-1,fj,fk,center)
    right=lap_y_value(fi+1,fj,fk,center)
    down=lap_y_value(fi,fj-1,fk,center)
    up=lap_y_value(fi,fj+1,fk,center)
    back=lap_y_value(fi,fj,fk-1,center)
    front=lap_y_value(fi,fj,fk+1,center)
    return (left+right+down+up+back+front-6.0*center)/(GRID_LEN*GRID_LEN)

@ti.func
def laplacian_z(fi:int,fj:int,fk:int)->ti.f32:
    center=grid_velz[face_z_id(fi,fj,fk)]
    left=lap_z_value(fi-1,fj,fk,center)
    right=lap_z_value(fi+1,fj,fk,center)
    down=lap_z_value(fi,fj-1,fk,center)
    up=lap_z_value(fi,fj+1,fk,center)
    back=lap_z_value(fi,fj,fk-1,center)
    front=lap_z_value(fi,fj,fk+1,center)
    return (left+right+down+up+back+front-6.0*center)/(GRID_LEN*GRID_LEN)

@ti.func
def sample_x(pos:vec3,vel:ti.template())->ti.f32:
    ans=0.0
    wsum=0.0
    base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN),ti.i32)
    base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
    base_k=ti.cast(ti.floor((pos[2]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
    for di,dj,dk in ti.ndrange((-1,3),(-1,3),(-1,3)):
        fi=base_i+di
        fj=base_j+dj
        fk=base_k+dk
        if fi>=0 and fi<=GRID_SIZE and fj>=0 and fj<GRID_SIZE and fk>=0 and fk<GRID_SIZE:
            fid=face_x_id(fi,fj,fk)
            if grid_validx[fid]==1:
                w=weight(face_x_pos(fi,fj,fk)-pos)
                ans+=vel[fid]*w
                wsum+=w
    if wsum>EPS:
        ans/=wsum
    return ans

@ti.func
def sample_y(pos:vec3,vel:ti.template())->ti.f32:
    ans=0.0
    wsum=0.0
    base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
    base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN),ti.i32)
    base_k=ti.cast(ti.floor((pos[2]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
    for di,dj,dk in ti.ndrange((-1,3),(-1,3),(-1,3)):
        fi=base_i+di
        fj=base_j+dj
        fk=base_k+dk
        if fi>=0 and fi<GRID_SIZE and fj>=0 and fj<=GRID_SIZE and fk>=0 and fk<GRID_SIZE:
            fid=face_y_id(fi,fj,fk)
            if grid_validy[fid]==1:
                w=weight(face_y_pos(fi,fj,fk)-pos)
                ans+=vel[fid]*w
                wsum+=w
    if wsum>EPS:
        ans/=wsum
    return ans

@ti.func
def sample_z(pos:vec3,vel:ti.template())->ti.f32:
    ans=0.0
    wsum=0.0
    base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
    base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
    base_k=ti.cast(ti.floor((pos[2]-DOMAIN_MIN)/GRID_LEN),ti.i32)
    for di,dj,dk in ti.ndrange((-1,3),(-1,3),(-1,3)):
        fi=base_i+di
        fj=base_j+dj
        fk=base_k+dk
        if fi>=0 and fi<GRID_SIZE and fj>=0 and fj<GRID_SIZE and fk>=0 and fk<=GRID_SIZE:
            fid=face_z_id(fi,fj,fk)
            if grid_validz[fid]==1:
                w=weight(face_z_pos(fi,fj,fk)-pos)
                ans+=vel[fid]*w
                wsum+=w
    if wsum>EPS:
        ans/=wsum
    return ans

@ti.func
def sample_vel(pos:vec3,velx:ti.template(),vely:ti.template(),velz:ti.template())->vec3:
    return ti.Vector([sample_x(pos,velx),sample_y(pos,vely),sample_z(pos,velz)])

@ti.kernel
def init_box_lines():
    p000=ti.Vector([DOMAIN_MIN,DOMAIN_MIN,DOMAIN_MIN])
    p100=ti.Vector([DOMAIN_MAX,DOMAIN_MIN,DOMAIN_MIN])
    p010=ti.Vector([DOMAIN_MIN,DOMAIN_MAX,DOMAIN_MIN])
    p110=ti.Vector([DOMAIN_MAX,DOMAIN_MAX,DOMAIN_MIN])
    p001=ti.Vector([DOMAIN_MIN,DOMAIN_MIN,DOMAIN_MAX])
    p101=ti.Vector([DOMAIN_MAX,DOMAIN_MIN,DOMAIN_MAX])
    p011=ti.Vector([DOMAIN_MIN,DOMAIN_MAX,DOMAIN_MAX])
    p111=ti.Vector([DOMAIN_MAX,DOMAIN_MAX,DOMAIN_MAX])
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
def init_grid_lines():
    n=GRID_SIZE+1
    offset_y=2*n*n
    offset_z=4*n*n
    for j,k in ti.ndrange(GRID_SIZE+1,GRID_SIZE+1):
        y=DOMAIN_MIN+ti.cast(j,ti.f32)*GRID_LEN
        z=DOMAIN_MIN+ti.cast(k,ti.f32)*GRID_LEN
        base=(j*n+k)*2
        grid_lines[base]=ti.Vector([DOMAIN_MIN,y,z])
        grid_lines[base+1]=ti.Vector([DOMAIN_MAX,y,z])
    for i,k in ti.ndrange(GRID_SIZE+1,GRID_SIZE+1):
        x=DOMAIN_MIN+ti.cast(i,ti.f32)*GRID_LEN
        z=DOMAIN_MIN+ti.cast(k,ti.f32)*GRID_LEN
        base=offset_y+(i*n+k)*2
        grid_lines[base]=ti.Vector([x,DOMAIN_MIN,z])
        grid_lines[base+1]=ti.Vector([x,DOMAIN_MAX,z])
    for i,j in ti.ndrange(GRID_SIZE+1,GRID_SIZE+1):
        x=DOMAIN_MIN+ti.cast(i,ti.f32)*GRID_LEN
        y=DOMAIN_MIN+ti.cast(j,ti.f32)*GRID_LEN
        base=offset_z+(i*n+j)*2
        grid_lines[base]=ti.Vector([x,y,DOMAIN_MIN])
        grid_lines[base+1]=ti.Vector([x,y,DOMAIN_MAX])

@ti.kernel
def init_particles():
    max_speed[None]=1e-5
    vel_limit_count[None]=0
    for i in range(NUM_PARTICLES):
        cell=i//PARTICLES_PER_CELL
        local=i%PARTICLES_PER_CELL
        cell_i=cell%LIQUID_CELL_X
        cell_j=(cell//LIQUID_CELL_X)%LIQUID_CELL_Y
        cell_k=cell//(LIQUID_CELL_X*LIQUID_CELL_Y)
        local_i=local%PARTICLES_PER_CELL_AXIS
        local_j=(local//PARTICLES_PER_CELL_AXIS)%PARTICLES_PER_CELL_AXIS
        local_k=local//(PARTICLES_PER_CELL_AXIS*PARTICLES_PER_CELL_AXIS)
        jitter_x=0.05*(ti.cast((i*17+13)%29,ti.f32)/29.0-0.5)
        jitter_y=0.05*(ti.cast((i*31+7)%37,ti.f32)/37.0-0.5)
        jitter_z=0.05*(ti.cast((i*19+11)%31,ti.f32)/31.0-0.5)
        offset_x=(ti.cast(local_i,ti.f32)+0.5+jitter_x)/ti.cast(PARTICLES_PER_CELL_AXIS,ti.f32)
        offset_y=(ti.cast(local_j,ti.f32)+0.5+jitter_y)/ti.cast(PARTICLES_PER_CELL_AXIS,ti.f32)
        offset_z=(ti.cast(local_k,ti.f32)+0.5+jitter_z)/ti.cast(PARTICLES_PER_CELL_AXIS,ti.f32)
        x=DOMAIN_MIN+(ti.cast(LIQUID_START_I+cell_i,ti.f32)+offset_x)*GRID_LEN
        y=DOMAIN_MIN+(ti.cast(LIQUID_START_J+cell_j,ti.f32)+offset_y)*GRID_LEN
        z=DOMAIN_MIN+(ti.cast(LIQUID_START_K+cell_k,ti.f32)+offset_z)*GRID_LEN
        parts_pos[i]=ti.Vector([x,y,z])
        parts_vel[i]=ti.Vector([0.0,0.0,0.0])
        old_parts_vel[i]=ti.Vector([0.0,0.0,0.0])
        new_part_vel[i]=ti.Vector([0.0,0.0,0.0])
        parts_grid[i]=part_cell(parts_pos[i])
    for i in range(FACE_X_NUM):
        grid_velx[i]=0.0
        tmp_grid_velx[i]=0.0
        old_grid_vel_x[i]=0.0
        grid_weightx[i]=0.0
        grid_validx[i]=0
    for i in range(FACE_Y_NUM):
        grid_vely[i]=0.0
        tmp_grid_vely[i]=0.0
        old_grid_vel_y[i]=0.0
        grid_weighty[i]=0.0
        grid_validy[i]=0
    for i in range(FACE_Z_NUM):
        grid_velz[i]=0.0
        tmp_grid_velz[i]=0.0
        old_grid_vel_z[i]=0.0
        grid_weightz[i]=0.0
        grid_validz[i]=0
    for i in range(CELL_NUM):
        grid_type[i]=TYPE_EMPTY
        grid_cnt[i]=0
        grid_solve[i]=0
        grid_pres[i]=0.0
        grid_density[i]=0.0

@ti.kernel
def compute_density():
    for i in range(CELL_NUM):
        grid_density[i]=0.0
    for p in range(NUM_PARTICLES):
        pos=parts_pos[p]
        base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        base_k=ti.cast(ti.floor((pos[2]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        for di,dj,dk in ti.ndrange((-1,3),(-1,3),(-1,3)):
            ci=base_i+di
            cj=base_j+dj
            ck=base_k+dk
            if ci>=0 and ci<GRID_SIZE and cj>=0 and cj<GRID_SIZE and ck>=0 and ck<GRID_SIZE:
                w=weight(cell_pos(ci,cj,ck)-pos)
                ti.atomic_add(grid_density[cell_id(ci,cj,ck)],w)

@ti.kernel
def init_rest_density():
    density_sum[None]=0.0
    density_cnt[None]=0.0
    for i in range(CELL_NUM):
        if grid_density[i]>DENSITY_REST_THRESHOLD:
            ti.atomic_add(density_sum[None],grid_density[i])
            ti.atomic_add(density_cnt[None],1.0)
    rest_density[None]=1.0
    if density_cnt[None]>0.0:
        rest_density[None]=density_sum[None]/density_cnt[None]

@ti.kernel
def P2G():
    for i,j,k in ti.ndrange(GRID_SIZE,GRID_SIZE,GRID_SIZE):
        gid=cell_id(i,j,k)
        grid_cnt[gid]=0
        grid_solve[gid]=0
        grid_pres[gid]=0.0
        grid_density[gid]=0.0
        if i==0 or i==GRID_SIZE-1 or j==0 or j==GRID_SIZE-1 or k==0 or k==GRID_SIZE-1:
            grid_type[gid]=TYPE_SOLID
        else:
            grid_type[gid]=TYPE_EMPTY
    for i in range(FACE_X_NUM):
        grid_velx[i]=0.0
        tmp_grid_velx[i]=0.0
        grid_weightx[i]=0.0
        grid_validx[i]=0
    for i in range(FACE_Y_NUM):
        grid_vely[i]=0.0
        tmp_grid_vely[i]=0.0
        grid_weighty[i]=0.0
        grid_validy[i]=0
    for i in range(FACE_Z_NUM):
        grid_velz[i]=0.0
        tmp_grid_velz[i]=0.0
        grid_weightz[i]=0.0
        grid_validz[i]=0
    for p in range(NUM_PARTICLES):
        old_parts_vel[p]=parts_vel[p]
        gid=part_cell(parts_pos[p])
        parts_grid[p]=gid
        if grid_type[gid]!=TYPE_SOLID:
            grid_type[gid]=TYPE_LIQUID
            ti.atomic_add(grid_cnt[gid],1)
    for p in range(NUM_PARTICLES):
        pos=parts_pos[p]
        base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        base_k=ti.cast(ti.floor((pos[2]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        for di,dj,dk in ti.ndrange((-1,3),(-1,3),(-1,3)):
            ci=base_i+di
            cj=base_j+dj
            ck=base_k+dk
            if ci>=0 and ci<GRID_SIZE and cj>=0 and cj<GRID_SIZE and ck>=0 and ck<GRID_SIZE:
                w=weight(cell_pos(ci,cj,ck)-pos)
                ti.atomic_add(grid_density[cell_id(ci,cj,ck)],w)
    for i,j,k in ti.ndrange(GRID_SIZE,GRID_SIZE,GRID_SIZE):
        gid=cell_id(i,j,k)
        if grid_type[gid]!=TYPE_SOLID:
            if grid_density[gid]>DENSITY_LIQUID_RATIO*rest_density[None]:
                grid_type[gid]=TYPE_LIQUID
    for i,j,k in ti.ndrange(GRID_SIZE,GRID_SIZE,GRID_SIZE):
        gid=cell_id(i,j,k)
        if grid_type[gid]==TYPE_LIQUID:
            if liquid_neighbor_count(i,j,k)>=MIN_SOLVE_NEIGHBORS:
                grid_solve[gid]=1
    for p in range(NUM_PARTICLES):
        pos=parts_pos[p]
        base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN),ti.i32)
        base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        base_k=ti.cast(ti.floor((pos[2]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        for di,dj,dk in ti.ndrange((-1,3),(-1,3),(-1,3)):
            fi=base_i+di
            fj=base_j+dj
            fk=base_k+dk
            if fi>=0 and fi<=GRID_SIZE and fj>=0 and fj<GRID_SIZE and fk>=0 and fk<GRID_SIZE:
                w=weight(face_x_pos(fi,fj,fk)-pos)
                fid=face_x_id(fi,fj,fk)
                ti.atomic_add(grid_velx[fid],parts_vel[p][0]*w)
                ti.atomic_add(grid_weightx[fid],w)
        base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN),ti.i32)
        base_k=ti.cast(ti.floor((pos[2]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        for di,dj,dk in ti.ndrange((-1,3),(-1,3),(-1,3)):
            fi=base_i+di
            fj=base_j+dj
            fk=base_k+dk
            if fi>=0 and fi<GRID_SIZE and fj>=0 and fj<=GRID_SIZE and fk>=0 and fk<GRID_SIZE:
                w=weight(face_y_pos(fi,fj,fk)-pos)
                fid=face_y_id(fi,fj,fk)
                ti.atomic_add(grid_vely[fid],parts_vel[p][1]*w)
                ti.atomic_add(grid_weighty[fid],w)
        base_i=ti.cast(ti.floor((pos[0]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        base_j=ti.cast(ti.floor((pos[1]-DOMAIN_MIN)/GRID_LEN-0.5),ti.i32)
        base_k=ti.cast(ti.floor((pos[2]-DOMAIN_MIN)/GRID_LEN),ti.i32)
        for di,dj,dk in ti.ndrange((-1,3),(-1,3),(-1,3)):
            fi=base_i+di
            fj=base_j+dj
            fk=base_k+dk
            if fi>=0 and fi<GRID_SIZE and fj>=0 and fj<GRID_SIZE and fk>=0 and fk<=GRID_SIZE:
                w=weight(face_z_pos(fi,fj,fk)-pos)
                fid=face_z_id(fi,fj,fk)
                ti.atomic_add(grid_velz[fid],parts_vel[p][2]*w)
                ti.atomic_add(grid_weightz[fid],w)
    for i in range(FACE_X_NUM):
        fi=i//(GRID_SIZE*GRID_SIZE)
        rem=i%(GRID_SIZE*GRID_SIZE)
        fj=rem//GRID_SIZE
        fk=rem%GRID_SIZE
        if has_liquid_x(fi,fj,fk)==1 and fi>0 and fi<GRID_SIZE and solid_face_x(fi,fj,fk)==0:
            grid_validx[i]=1
        if grid_weightx[i]>EPS:
            grid_velx[i]/=grid_weightx[i]
        if fi==0 or fi==GRID_SIZE:
            grid_velx[i]=0.0
        old_grid_vel_x[i]=grid_velx[i]
    for i in range(FACE_Y_NUM):
        fi=i//((GRID_SIZE+1)*GRID_SIZE)
        rem=i%((GRID_SIZE+1)*GRID_SIZE)
        fj=rem//GRID_SIZE
        fk=rem%GRID_SIZE
        if has_liquid_y(fi,fj,fk)==1 and fj>0 and fj<GRID_SIZE and solid_face_y(fi,fj,fk)==0:
            grid_validy[i]=1
        if grid_weighty[i]>EPS:
            grid_vely[i]/=grid_weighty[i]
        if fj==0 or fj==GRID_SIZE:
            grid_vely[i]=0.0
        old_grid_vel_y[i]=grid_vely[i]
    for i in range(FACE_Z_NUM):
        fi=i//(GRID_SIZE*(GRID_SIZE+1))
        rem=i%(GRID_SIZE*(GRID_SIZE+1))
        fj=rem//(GRID_SIZE+1)
        fk=rem%(GRID_SIZE+1)
        if has_liquid_z(fi,fj,fk)==1 and fk>0 and fk<GRID_SIZE and solid_face_z(fi,fj,fk)==0:
            grid_validz[i]=1
        if grid_weightz[i]>EPS:
            grid_velz[i]/=grid_weightz[i]
        if fk==0 or fk==GRID_SIZE:
            grid_velz[i]=0.0
        old_grid_vel_z[i]=grid_velz[i]

@ti.kernel
def add_force():
    nu=ETA/RHO
    for i in range(FACE_X_NUM):
        fi=i//(GRID_SIZE*GRID_SIZE)
        rem=i%(GRID_SIZE*GRID_SIZE)
        fj=rem//GRID_SIZE
        fk=rem%GRID_SIZE
        tmp_grid_velx[i]=grid_velx[i]
        if grid_validx[i]==1:
            tmp_grid_velx[i]=grid_velx[i]+dt[None]*nu*laplacian_x(fi,fj,fk)
    for i in range(FACE_Y_NUM):
        fi=i//((GRID_SIZE+1)*GRID_SIZE)
        rem=i%((GRID_SIZE+1)*GRID_SIZE)
        fj=rem//GRID_SIZE
        fk=rem%GRID_SIZE
        tmp_grid_vely[i]=grid_vely[i]
        if grid_validy[i]==1:
            tmp_grid_vely[i]=grid_vely[i]+dt[None]*(GRAVITY+nu*laplacian_y(fi,fj,fk))
    for i in range(FACE_Z_NUM):
        fi=i//(GRID_SIZE*(GRID_SIZE+1))
        rem=i%(GRID_SIZE*(GRID_SIZE+1))
        fj=rem//(GRID_SIZE+1)
        fk=rem%(GRID_SIZE+1)
        tmp_grid_velz[i]=grid_velz[i]
        if grid_validz[i]==1:
            tmp_grid_velz[i]=grid_velz[i]+dt[None]*nu*laplacian_z(fi,fj,fk)
    for i in range(FACE_X_NUM):
        grid_velx[i]=tmp_grid_velx[i]
    for i in range(FACE_Y_NUM):
        grid_vely[i]=tmp_grid_vely[i]
    for i in range(FACE_Z_NUM):
        grid_velz[i]=tmp_grid_velz[i]

@ti.kernel
def solve_pressure():
    coef=RHO*GRID_LEN*GRID_LEN/dt[None]
    wall_coef=RHO*GRID_LEN/dt[None]
    for _ in range(PRESSURE_ITERS):
        for i,j,k in ti.ndrange(GRID_SIZE,GRID_SIZE,GRID_SIZE):
            gid=cell_id(i,j,k)
            if grid_solve[gid]==1:
                div=(grid_velx[face_x_id(i+1,j,k)]-grid_velx[face_x_id(i,j,k)]+grid_vely[face_y_id(i,j+1,k)]-grid_vely[face_y_id(i,j,k)]+grid_velz[face_z_id(i,j,k+1)]-grid_velz[face_z_id(i,j,k)])/GRID_LEN
                target_div=target_divergence(gid)
                cnt=0.0
                psum=0.0
                solid_term=0.0
                if is_solid_cell(i-1,j,k)==0:
                    cnt+=1.0
                    psum+=pressure_at(i-1,j,k)
                else:
                    solid_term+=-wall_coef*grid_velx[face_x_id(i,j,k)]
                if is_solid_cell(i+1,j,k)==0:
                    cnt+=1.0
                    psum+=pressure_at(i+1,j,k)
                else:
                    solid_term+=wall_coef*grid_velx[face_x_id(i+1,j,k)]
                if is_solid_cell(i,j-1,k)==0:
                    cnt+=1.0
                    psum+=pressure_at(i,j-1,k)
                else:
                    solid_term+=-wall_coef*grid_vely[face_y_id(i,j,k)]
                if is_solid_cell(i,j+1,k)==0:
                    cnt+=1.0
                    psum+=pressure_at(i,j+1,k)
                else:
                    solid_term+=wall_coef*grid_vely[face_y_id(i,j+1,k)]
                if is_solid_cell(i,j,k-1)==0:
                    cnt+=1.0
                    psum+=pressure_at(i,j,k-1)
                else:
                    solid_term+=-wall_coef*grid_velz[face_z_id(i,j,k)]
                if is_solid_cell(i,j,k+1)==0:
                    cnt+=1.0
                    psum+=pressure_at(i,j,k+1)
                else:
                    solid_term+=wall_coef*grid_velz[face_z_id(i,j,k+1)]
                if cnt>0.0:
                    grid_pres[gid]=(psum+solid_term-coef*(div-target_div))/cnt

@ti.kernel
def project_velocity():
    scale=dt[None]/(RHO*GRID_LEN)
    for i in range(FACE_X_NUM):
        fi=i//(GRID_SIZE*GRID_SIZE)
        rem=i%(GRID_SIZE*GRID_SIZE)
        fj=rem//GRID_SIZE
        fk=rem%GRID_SIZE
        if fi==0 or fi==GRID_SIZE:
            grid_velx[i]=0.0
        elif solid_face_x(fi,fj,fk)==1:
            grid_velx[i]=0.0
        elif is_pressure_cell(fi-1,fj,fk)==1 or is_pressure_cell(fi,fj,fk)==1:
            grid_velx[i]-=scale*(pressure_at(fi,fj,fk)-pressure_at(fi-1,fj,fk))
        elif has_liquid_x(fi,fj,fk)==0:
            grid_velx[i]=0.0
    for i in range(FACE_Y_NUM):
        fi=i//((GRID_SIZE+1)*GRID_SIZE)
        rem=i%((GRID_SIZE+1)*GRID_SIZE)
        fj=rem//GRID_SIZE
        fk=rem%GRID_SIZE
        if fj==0 or fj==GRID_SIZE:
            grid_vely[i]=0.0
        elif solid_face_y(fi,fj,fk)==1:
            grid_vely[i]=0.0
        elif is_pressure_cell(fi,fj-1,fk)==1 or is_pressure_cell(fi,fj,fk)==1:
            grid_vely[i]-=scale*(pressure_at(fi,fj,fk)-pressure_at(fi,fj-1,fk))
        elif has_liquid_y(fi,fj,fk)==0:
            grid_vely[i]=0.0
    for i in range(FACE_Z_NUM):
        fi=i//(GRID_SIZE*(GRID_SIZE+1))
        rem=i%(GRID_SIZE*(GRID_SIZE+1))
        fj=rem//(GRID_SIZE+1)
        fk=rem%(GRID_SIZE+1)
        if fk==0 or fk==GRID_SIZE:
            grid_velz[i]=0.0
        elif solid_face_z(fi,fj,fk)==1:
            grid_velz[i]=0.0
        elif is_pressure_cell(fi,fj,fk-1)==1 or is_pressure_cell(fi,fj,fk)==1:
            grid_velz[i]-=scale*(pressure_at(fi,fj,fk)-pressure_at(fi,fj,fk-1))
        elif has_liquid_z(fi,fj,fk)==0:
            grid_velz[i]=0.0

@ti.kernel
def extrapolate_velocity():
    for _ in ti.static(range(EXTRAPOLATE_ITERS)):
        for i in range(FACE_X_NUM):
            fi=i//(GRID_SIZE*GRID_SIZE)
            rem=i%(GRID_SIZE*GRID_SIZE)
            fj=rem//GRID_SIZE
            fk=rem%GRID_SIZE
            tmp_grid_velx[i]=grid_velx[i]
            tmp_validx[i]=grid_validx[i]
            if grid_validx[i]==0 and fi>0 and fi<GRID_SIZE and solid_face_x(fi,fj,fk)==0:
                total=0.0
                cnt=0.0
                for di,dj,dk in ti.ndrange((-1,2),(-1,2),(-1,2)):
                    if ti.abs(di)+ti.abs(dj)+ti.abs(dk)==1:
                        ni=fi+di
                        nj=fj+dj
                        nk=fk+dk
                        if ni>=0 and ni<=GRID_SIZE and nj>=0 and nj<GRID_SIZE and nk>=0 and nk<GRID_SIZE:
                            nid=face_x_id(ni,nj,nk)
                            if grid_validx[nid]==1:
                                total+=grid_velx[nid]
                                cnt+=1.0
                if cnt>0.0:
                    tmp_grid_velx[i]=total/cnt
                    tmp_validx[i]=1
        for i in range(FACE_X_NUM):
            if grid_validx[i]==0 and tmp_validx[i]==1:
                old_grid_vel_x[i]=tmp_grid_velx[i]
            grid_velx[i]=tmp_grid_velx[i]
            grid_validx[i]=tmp_validx[i]
        for i in range(FACE_Y_NUM):
            fi=i//((GRID_SIZE+1)*GRID_SIZE)
            rem=i%((GRID_SIZE+1)*GRID_SIZE)
            fj=rem//GRID_SIZE
            fk=rem%GRID_SIZE
            tmp_grid_vely[i]=grid_vely[i]
            tmp_validy[i]=grid_validy[i]
            if grid_validy[i]==0 and fj>0 and fj<GRID_SIZE and solid_face_y(fi,fj,fk)==0:
                total=0.0
                cnt=0.0
                for di,dj,dk in ti.ndrange((-1,2),(-1,2),(-1,2)):
                    if ti.abs(di)+ti.abs(dj)+ti.abs(dk)==1:
                        ni=fi+di
                        nj=fj+dj
                        nk=fk+dk
                        if ni>=0 and ni<GRID_SIZE and nj>=0 and nj<=GRID_SIZE and nk>=0 and nk<GRID_SIZE:
                            nid=face_y_id(ni,nj,nk)
                            if grid_validy[nid]==1:
                                total+=grid_vely[nid]
                                cnt+=1.0
                if cnt>0.0:
                    tmp_grid_vely[i]=total/cnt
                    tmp_validy[i]=1
        for i in range(FACE_Y_NUM):
            if grid_validy[i]==0 and tmp_validy[i]==1:
                old_grid_vel_y[i]=tmp_grid_vely[i]
            grid_vely[i]=tmp_grid_vely[i]
            grid_validy[i]=tmp_validy[i]
        for i in range(FACE_Z_NUM):
            fi=i//(GRID_SIZE*(GRID_SIZE+1))
            rem=i%(GRID_SIZE*(GRID_SIZE+1))
            fj=rem//(GRID_SIZE+1)
            fk=rem%(GRID_SIZE+1)
            tmp_grid_velz[i]=grid_velz[i]
            tmp_validz[i]=grid_validz[i]
            if grid_validz[i]==0 and fk>0 and fk<GRID_SIZE and solid_face_z(fi,fj,fk)==0:
                total=0.0
                cnt=0.0
                for di,dj,dk in ti.ndrange((-1,2),(-1,2),(-1,2)):
                    if ti.abs(di)+ti.abs(dj)+ti.abs(dk)==1:
                        ni=fi+di
                        nj=fj+dj
                        nk=fk+dk
                        if ni>=0 and ni<GRID_SIZE and nj>=0 and nj<GRID_SIZE and nk>=0 and nk<=GRID_SIZE:
                            nid=face_z_id(ni,nj,nk)
                            if grid_validz[nid]==1:
                                total+=grid_velz[nid]
                                cnt+=1.0
                if cnt>0.0:
                    tmp_grid_velz[i]=total/cnt
                    tmp_validz[i]=1
        for i in range(FACE_Z_NUM):
            if grid_validz[i]==0 and tmp_validz[i]==1:
                old_grid_vel_z[i]=tmp_grid_velz[i]
            grid_velz[i]=tmp_grid_velz[i]
            grid_validz[i]=tmp_validz[i]

@ti.kernel
def G2P():
    for i in range(NUM_PARTICLES):
        pos=parts_pos[i]
        gid=parts_grid[i]
        pic=sample_vel(pos,grid_velx,grid_vely,grid_velz)
        old_grid=sample_vel(pos,old_grid_vel_x,old_grid_vel_y,old_grid_vel_z)
        flip=old_parts_vel[i]+pic-old_grid
        ratio=alpha[None]
        if grid_solve[gid]==0:
            ratio=ti.min(ratio,SURFACE_ALPHA)
        if pos[0]<WALL_MIN+GRID_LEN or pos[0]>WALL_MAX-GRID_LEN:
            ratio=ti.min(ratio,SURFACE_ALPHA)
        if pos[1]<WALL_MIN+GRID_LEN or pos[1]>WALL_MAX-GRID_LEN:
            ratio=ti.min(ratio,SURFACE_ALPHA)
        if pos[2]<WALL_MIN+GRID_LEN or pos[2]>WALL_MAX-GRID_LEN:
            ratio=ti.min(ratio,SURFACE_ALPHA)
        new_part_vel[i]=(1.0-ratio)*pic+ratio*flip

@ti.kernel
def update_pos():
    lo=WALL_MIN+PART_RADIUS
    hi=WALL_MAX-PART_RADIUS
    max_speed[None]=1e-5
    vel_limit_count[None]=0
    for i in range(NUM_PARTICLES):
        vel=new_part_vel[i]
        speed=vel.norm()
        if speed>MAX_PARTICLE_SPEED:
            vel=vel/speed*MAX_PARTICLE_SPEED
            speed=MAX_PARTICLE_SPEED
            ti.atomic_add(vel_limit_count[None],1)
        pos=parts_pos[i]+vel*dt[None]
        if pos[0]<lo:
            pos[0]=lo
            if vel[0]<0.0:
                vel[0]=-vel[0]*BOUNCE
            vel[1]*=WALL_TANGENT_DAMP
            vel[2]*=WALL_TANGENT_DAMP
        if pos[0]>hi:
            pos[0]=hi
            if vel[0]>0.0:
                vel[0]=-vel[0]*BOUNCE
            vel[1]*=WALL_TANGENT_DAMP
            vel[2]*=WALL_TANGENT_DAMP
        if pos[1]<lo:
            pos[1]=lo
            if vel[1]<0.0:
                vel[1]=-vel[1]*BOUNCE
            vel[0]*=WALL_TANGENT_DAMP
            vel[2]*=WALL_TANGENT_DAMP
        if pos[1]>hi:
            pos[1]=hi
            if vel[1]>0.0:
                vel[1]=-vel[1]*BOUNCE
            vel[0]*=WALL_TANGENT_DAMP
            vel[2]*=WALL_TANGENT_DAMP
        if pos[2]<lo:
            pos[2]=lo
            if vel[2]<0.0:
                vel[2]=-vel[2]*BOUNCE
            vel[0]*=WALL_TANGENT_DAMP
            vel[1]*=WALL_TANGENT_DAMP
        if pos[2]>hi:
            pos[2]=hi
            if vel[2]>0.0:
                vel[2]=-vel[2]*BOUNCE
            vel[0]*=WALL_TANGENT_DAMP
            vel[1]*=WALL_TANGENT_DAMP
        ti.atomic_max(max_speed[None],speed)
        parts_pos[i]=pos
        parts_vel[i]=vel

def substep():
    P2G()
    add_force()
    solve_pressure()
    project_velocity()
    extrapolate_velocity()
    G2P()
    update_pos()

def init():
    dt[None]=DT_INIT
    alpha[None]=0.95
    rest_density[None]=1.0
    pressure_abs_max[None]=EPS
    init_box_lines()
    init_grid_lines()
    init_particles()
    compute_density()
    init_rest_density()

def clamp(v,lo,hi):
    return max(lo,min(v,hi))

def init_camera(camera):
    camera.position(1.45,1.15,1.65)
    camera.lookat(0.50,0.45,0.50)
    camera.up(0.0,1.0,0.0)
    camera.fov(55)

def handle_input(window,camera):
    global paused
    for e in window.get_events(ti.ui.PRESS):
        if e.key==ti.ui.ESCAPE:
            window.running=False
        elif e.key==" ":
            paused=not paused
        elif e.key=="r" or e.key=="R":
            init()
        elif e.key=="c" or e.key=="C":
            init_camera(camera)
        elif e.key=="[":
            dt[None]=clamp(dt[None]*0.8,DT_MIN,DT_MAX)
        elif e.key=="]":
            dt[None]=clamp(dt[None]*1.25,DT_MIN,DT_MAX)
    if window.is_pressed(ti.ui.LEFT):
        alpha[None]=clamp(alpha[None]-0.01,0.0,1.0)
    if window.is_pressed(ti.ui.RIGHT):
        alpha[None]=clamp(alpha[None]+0.01,0.0,1.0)

def render(window,scene,camera):
    scene.set_camera(camera)
    scene.ambient_light((0.35,0.35,0.35))
    scene.point_light(pos=(1.4,1.6,1.2),color=(1.0,1.0,1.0))
    scene.lines(grid_lines,width=0.85,color=(0.38,0.42,0.48))
    scene.lines(box_lines,width=2.2,color=(0.78,0.82,0.88))
    scene.particles(parts_pos,radius=PART_RADIUS,color=(0.18,0.48,0.95))
    canvas=window.get_canvas()
    canvas.set_background_color((0.07,0.08,0.09))
    canvas.scene(scene)

def render_gui(window):
    gui=window.get_gui()
    gui.begin("3D IDP",0.02,0.02,0.25,0.20)
    gui.text(f"particles: {NUM_PARTICLES}")
    gui.text(f"grid: {GRID_SIZE}^3")
    gui.text(f"flipRatio: {alpha[None]:.2f}")
    alpha[None]=clamp(gui.slider_float("flipRatio",alpha[None],0.0,1.0),0.0,1.0)
    gui.text(f"IDP beta: {IDP_BETA:.2f}")
    gui.text(f"rest density: {rest_density[None]:.2f}")
    dt[None]=clamp(gui.slider_float("dt",dt[None],DT_MIN,DT_MAX),DT_MIN,DT_MAX)
    gui.text(f"max speed: {max_speed[None]:.3f}")
    gui.text(f"vel cap: {vel_limit_count[None]}")
    gui.end()

def main():
    init()
    window=ti.ui.Window("3D PIC/FLIP IDP",(1100,900))
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
