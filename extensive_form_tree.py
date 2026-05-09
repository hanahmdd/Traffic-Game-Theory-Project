import graphviz


def generate_game_tree():

    dot = graphviz.Digraph(
        'TrafficGameTree',
        comment='Traffic Extensive Form'
    )

    dot.attr(rankdir='TB', size='9,8')

    dot.attr(
        'node',
        shape='circle',
        style='filled',
        color='lightblue'
    )

    # ROOT
  

    dot.node(
        'root',
        'Start'
    )

    # AGENT A
  

    dot.node(
        'A',
        'Agent A',
        fillcolor='#FF9999'
    )

    dot.edge('root', 'A')

   
    # ACTIONS
    
    dot.node(
        'A_S',
        'Short',
        shape='box'
    )

    dot.node(
        'A_L',
        'Long',
        shape='box'
    )

    dot.edge(
        'A',
        'A_S',
        label='Short'
    )

    dot.edge(
        'A',
        'A_L',
        label='Long'
    )


    # OBSERVATIONS
 

    dot.node(
        'Noise_S',
        'Observed Short',
        shape='diamond',
        fillcolor='#FFF2CC'
    )

    dot.node(
        'Noise_L',
        'Observed Long',
        shape='diamond',
        fillcolor='#FFF2CC'
    )

    dot.edge('A_S', 'Noise_S')
    dot.edge('A_L', 'Noise_L')

  
    # SEPARATE DECISION NODES


    dot.node(
        'B1',
        'Agent B',
        fillcolor='#99FF99'
    )

    dot.node(
        'B2',
        'Agent B',
        fillcolor='#99FF99'
    )

    dot.edge('Noise_S', 'B1')
    dot.edge('Noise_L', 'B2')

   
    # INFORMATION SET


    dot.edge(
        'B1',
        'B2',
        style='dotted',
        dir='none',
        color='red',
        label='Information Set'
    )

    
    # PAYOFFS


    payoff_nodes = {

        'P1': '(5,5)',
        'P2': '(3,8)',
        'P3': '(8,3)',
        'P4': '(2,2)'
    }

    for key, value in payoff_nodes.items():

        dot.node(
            key,
            value,
            shape='plaintext',
            fontcolor='red'
        )

    dot.edge('B1', 'P1', label='Short')
    dot.edge('B1', 'P2', label='Long')

    dot.edge('B2', 'P3', label='Short')
    dot.edge('B2', 'P4', label='Long')

    print(
        "[Visualization] Generating improved extensive form tree..."
    )

    dot.render(
        'extensive_form_tree',
        format='png',
        cleanup=True
    )

    print(
        "[Success] extensive_form_tree.png saved."
    )


if __name__ == "__main__":
    generate_game_tree()