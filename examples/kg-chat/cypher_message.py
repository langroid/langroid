CONSTRUCT_DEPENDENCY_GRAPH = """
        with "{package_type}" as system, "{package_name}" as name, "{package_version}" as version

        call apoc.load.model_dump_json("https://api.deps.dev/v3alpha/systems/"+system+"/packages/"
                            +name+"/versions/"+version+":dependencies")
        yield value as r
        
        call {{ with r
                unwind r.nodes as package
                merge (p:Package:PyPi {{name: package.versionKey.name, version: package.versionKey.version}})
                return collect(p) as packages
        }}
        call {{ with r, packages
            unwind r.edges as edge
            with packages[edge.fromNode] as from, packages[edge.toNode] as to, edge
            merge (from)-[rel:DEPENDS_ON]->(to) ON CREATE SET rel.requirement 
            = edge.requirement
            return count(*) as numRels
        }}
        
        match (root:Package:PyPi) where root.imported is null
        set root.imported = true
        with "{package_type}" as system, root.name as name, root.version as version
        call apoc.load.model_dump_json("https://api.deps.dev/v3alpha/systems/"+system+"/packages/"
                            +name+"/versions/"+version+":dependencies")
        yield value as r
        
        call {{ with r
                unwind r.nodes as package
                merge (p:Package:PyPi {{name: package.versionKey.name, version: package.versionKey.version}})
                return collect(p) as packages
        }}
        call {{ with r, packages
                unwind r.edges as edge
                with packages[edge.fromNode] as from, packages[edge.toNode] as to, edge
                merge (from)-[rel:DEPENDS_ON]->(to) ON CREATE SET 
                rel.requirement = edge.requirement
                return count(*) as numRels
        }}
        return size(packages) as numPackages, numRels
        """
