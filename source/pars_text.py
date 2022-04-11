import re, MySQLdb, time


file = open('genes1.txt')
file = file.readlines()

gene_list = ['GATA3']
for line in file:
    line = line.strip().split()

    gene = line[1].strip(';').strip('"')
    gene_list.append(gene)


def id2ensembl(id_list,
               genus_name,
               species_name,
               ensemblver=None):
    ''' Converts a list of any IDs into ensemblIDs

        Requires a list of IDs, a genus and species name, and
        optionally the enembl version to use. Genus_name and
        species_name must be the same that ensembl uses.

        For each matching record the script returns a list of tuples containing:
        (original ID, database ID matches, ensembl gene ID, biotype, description,
        chromosome, start coord, stop coord, strand).

        Example:

        ensembl_genes = id2ensembl(["SCFD2", "PPAP2C", "LASS6"],
                                       genus_name="homo",
                                       species_name="sapiens")

        original published here: https://gist.github.com/3452151
        update published here: https://gist.github.com/3487753
        '''


    # ensemlb details
    ensembl_host = "ensembldb.ensembl.org"
    ensembl_port = 5306
    # open connection
    ensemblDB = MySQLdb.connect(host=ensembl_host,
                                user="anonymous",
                                port=ensembl_port)
    cur = ensemblDB.cursor()
    # build ensembl species string
    ensembl_species = "%s_%s" % (genus_name.lower(), species_name.lower())
    # get tables in the schema matching the query species
    schema_sql = 'show schemas like "%s_core%s_%s"' % (ensembl_species,
                                                       "\\",
                                                       "%")
    cur.execute(schema_sql)
    records = cur.fetchall()
    if ensemblver is None:
    # trim out the latest one and split the version number
    # and table version number
        latest_ver = records[-1][0].split("_")[-2:]
    else:
        for record in records:
            if re.match("%s_core_%i.+" % (ensembl_species,
                                          ensemblver), record[0]):
                latest_ver = record[0].split("_")[-2:]
    # define dbs to use
    core_db = "%s_core_%s_%s" % (ensembl_species,
                                 latest_ver[0],
                                 latest_ver[1])
    # id query string
    # 'external_db_id=1300' = EntrezGene
    query_str = '''
            SELECT 
                x.display_label AS ID,
                GROUP_CONCAT(xdb.db_display_name) AS IDmatches,
                IFNULL(g.stable_id, pg.stable_id) AS eGene,
                IFNULL(g.biotype, pg.biotype) AS type,
                IFNULL(g.description, pg.description) AS gdesc,
                c.name AS chr,
                IFNULL(g.seq_region_start, pg.seq_region_start) AS start,
                IFNULL(g.seq_region_end, pg.seq_region_end) AS end,
                IFNULL(g.seq_region_strand, pg.seq_region_strand) AS strand    
            FROM %s.xref x 
            JOIN %s.external_db xdb ON x.external_db_id=xdb.external_db_id 
            LEFT JOIN %s.dependent_xref dx ON x.xref_id=dx.dependent_xref_id 
            LEFT JOIN %s.object_xref ox ON dx.object_xref_id=ox.object_xref_id
            LEFT JOIN %s.gene g ON ox.ensembl_id=g.gene_id 
            LEFT JOIN %s.gene pg ON pg.display_xref_id=x.xref_id
            JOIN %s.seq_region c ON g.seq_region_id=c.seq_region_id
            JOIN %s.coord_system cs ON c.coord_system_id=cs.coord_system_id
            WHERE cs.rank=1
            AND x.display_label IN ("%s")
            GROUP BY eGene;
            ''' % (core_db, core_db, core_db,
                   core_db, core_db, core_db,
                   core_db, core_db, '", "'.join(id_list)
                   )
    # execute query
    ttime = time.gmtime()
    timestring = "%02.0d:%02.0d:%02.0d" % (ttime[3], ttime[4], ttime[5])
    print
    "querying the ensembl database for all ensembl genes at %s..." % timestring
    t1 = time.time()
    cur.execute(query_str)  # this is not so quick (~5min)
    records = cur.fetchall()
    print
    "returned %i records in %.2fs" % (len(records), (time.time() - t1))
    return (records)


ensembl_genes = id2ensembl(gene_list, genus_name='homo', species_name='sapiens')

ensembl_genes_list = [e[2] for i, e in enumerate(ensembl_genes)]

print(ensembl_genes_list)

with open('gene_list.txt', 'w') as output:
    #output.write(str(ensembl_genes_list))
    for gene in ensembl_genes_list:
        output.write('%s \n' %gene)



