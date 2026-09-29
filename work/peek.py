import polars as pl, time
t=time.time()
rd=lambda p: pl.read_csv(p,separator='\t',quote_char=None,infer_schema=False,encoding='utf8-lossy')
s1=rd('dataset/train/train_source1.tsv'); s2=rd('dataset/train/train_source2.tsv'); s3=rd('dataset/train/train_source3.tsv')
gt=rd('dataset/train/train_ground_truth.tsv')
print('load',time.time()-t, s1.shape,s2.shape,s3.shape)
o=pl.concat([s2,s3])
g=gt.head(4000).tail(12).with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids')
for sid,grp in g.group_by('source1_entity_id',maintain_order=True):
    r=s1.filter(pl.col('entity_id')==sid[0]).row(0)
    print('\n###',r[1],'|',r[2],'|',r[3])
    ids=grp['matched_entity_ids'].to_list()
    for x in o.filter(pl.col('entity_id').is_in(ids)).iter_rows(): print('   ',x[0][:2],x[1],'|',x[2])
