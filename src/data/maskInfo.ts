// 한국 전통 탈 정보 데이터베이스
export interface MaskDetailInfo {
  name: string;
  koreanName: string;
  origin: string;
  description: string;
  danceDescription: string;
  character?: string;
  videoUrl?: string;
  wikiUrl?: string;
}

export const MASK_INFO: Record<string, MaskDetailInfo> = {
  // 하회탈 (안동 하회마을)
  yangban: {
    name: 'yangban',
    koreanName: '양반',
    origin: '안동 하회탈',
    description: '양반 계층을 풍자하는 탈로, 턱이 분리되어 대사에 따라 움직입니다. 권위적이면서도 우스꽝스러운 표정이 특징입니다.',
    danceDescription: '하회별신굿탈놀이에서 양반은 선비를 데리고 다니며 백정에게 우롱당하는 모습을 보여줍니다. 유교적 질서를 풍자하는 대표적인 장면입니다.',
    character: '위선적이고 권위적인 양반 계층의 허세를 풍자',
    videoUrl: 'https://www.youtube.com/watch?v=ZxQYKwGhwCY',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  bune: {
    name: 'bune',
    koreanName: '부네',
    origin: '안동 하회탈',
    description: '젊은 여인을 상징하는 탈로, 아름다운 미소와 부드러운 곡선이 특징입니다. 한국 전통 탈 중 가장 아름다운 여성상으로 평가받습니다.',
    danceDescription: '부네는 중의 유혹을 받는 장면에서 등장하며, 양반과 선비가 부네를 두고 다투는 모습을 통해 당시 사회상을 보여줍니다.',
    character: '순수하고 아름다운 젊은 여인',
    videoUrl: 'https://www.youtube.com/watch?v=ZxQYKwGhwCY',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  gaksi: {
    name: 'gaksi',
    koreanName: '각시',
    origin: '안동 하회탈',
    description: '신부 또는 새색시를 상징하는 탈입니다. 부네보다 더 어리고 수줍은 표정을 하고 있습니다.',
    danceDescription: '각시탈은 혼례와 관련된 장면에서 주로 등장하며, 순결하고 정숙한 여인상을 표현합니다.',
    character: '수줍고 정숙한 새색시',
    videoUrl: 'https://www.youtube.com/watch?v=ZxQYKwGhwCY',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  jung: {
    name: 'jung',
    koreanName: '중',
    origin: '안동 하회탈',
    description: '파계승을 상징하는 탈로, 탐욕스럽고 음탕한 표정이 특징입니다. 불교 승려의 타락상을 풍자합니다.',
    danceDescription: '중은 부네를 유혹하는 장면에서 등장하며, 파계승의 위선적인 모습을 해학적으로 표현합니다.',
    character: '탐욕스럽고 위선적인 파계승',
    videoUrl: 'https://www.youtube.com/watch?v=ZxQYKwGhwCY',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  imae: {
    name: 'imae',
    koreanName: '이매',
    origin: '안동 하회탈',
    description: '바보 또는 천치를 상징하는 탈로, 한쪽 눈이 찌그러지고 입이 비뚤어진 비대칭 얼굴이 특징입니다.',
    danceDescription: '이매는 어리숙하지만 순수한 캐릭터로, 양반의 허세를 본의 아니게 폭로하는 역할을 합니다.',
    character: '순수하고 어리숙한 바보',
    videoUrl: 'https://www.youtube.com/watch?v=ZxQYKwGhwCY',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  chorangi: {
    name: 'chorangi',
    koreanName: '초랭이',
    origin: '안동 하회탈',
    description: '양반의 하인을 상징하는 탈로, 재치 있고 영리한 표정이 특징입니다. 주인을 놀리면서도 충직한 하인의 모습을 보여줍니다.',
    danceDescription: '초랭이는 양반을 수행하며 그의 허세를 꼬집는 역할을 합니다. 민중의 지혜와 해학을 대변합니다.',
    character: '영리하고 재치 있는 하인',
    videoUrl: 'https://www.youtube.com/watch?v=ZxQYKwGhwCY',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  baekjeong: {
    name: 'baekjeong',
    koreanName: '백정',
    origin: '안동 하회탈',
    description: '도살업자를 상징하는 탈로, 투박하고 거친 인상이 특징입니다. 천민 계층이지만 실제 능력이 있는 인물을 표현합니다.',
    danceDescription: '백정은 소의 생식기를 들고 양반을 조롱하는 장면에서 등장하며, 신분 질서에 대한 통렬한 풍자를 보여줍니다.',
    character: '거칠지만 능력 있는 천민',
    videoUrl: 'https://www.youtube.com/watch?v=ZxQYKwGhwCY',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  halmi: {
    name: 'halmi',
    koreanName: '할미',
    origin: '안동 하회탈',
    description: '늙은 여인을 상징하는 탈로, 주름진 얼굴과 이빨 빠진 입이 특징입니다. 고단한 삶을 살아온 서민 여성을 표현합니다.',
    danceDescription: '할미는 억척스럽게 살아온 서민 여성의 애환을 표현하며, 관객들의 공감을 이끌어냅니다.',
    character: '고단하지만 강인한 노년 여성',
    videoUrl: 'https://www.youtube.com/watch?v=ZxQYKwGhwCY',
    wikiUrl: 'https://ko.wikipedia.org/wiki/하회탈'
  },
  
  // 봉산탈 (황해도 봉산)
  malttugi: {
    name: 'malttugi',
    koreanName: '말뚝이',
    origin: '봉산탈춤',
    description: '양반의 하인으로, 봉산탈춤에서 가장 인기 있는 캐릭터입니다. 해학적이고 풍자적인 대사로 양반을 조롱합니다.',
    danceDescription: '말뚝이는 채찍을 들고 등장하여 양반 삼형제를 신랄하게 비판하는 독백을 합니다. "쉬이~" 하는 소리와 함께 시작되는 그의 대사는 봉산탈춤의 하이라이트입니다.',
    character: '지혜롭고 풍자적인 하인',
    videoUrl: 'https://www.youtube.com/watch?v=EhGBPqGKBl0',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  miyal: {
    name: 'miyal',
    koreanName: '미얄',
    origin: '봉산탈춤',
    description: '영감의 본처로, 남편에게 버림받은 늙은 여인입니다. 서글프면서도 익살스러운 표정이 특징입니다.',
    danceDescription: '미얄할미는 영감과 첩 덜머리집 사이에서 갈등하다 결국 쫓겨나는 비극적 장면을 연기합니다. 조선시대 여성의 비애를 표현합니다.',
    character: '버림받은 슬픈 본처',
    videoUrl: 'https://www.youtube.com/watch?v=EhGBPqGKBl0',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  chwibari: {
    name: 'chwibari',
    koreanName: '취발이',
    origin: '봉산탈춤',
    description: '파계승을 상징하는 탈로, 머리카락이 헝클어진 모습이 특징입니다. 술에 취한 듯한 붉은 얼굴이 인상적입니다.',
    danceDescription: '취발이는 노장 과장에서 등장하여 소무를 유혹하는 파계승의 모습을 보여줍니다.',
    character: '타락한 파계승',
    videoUrl: 'https://www.youtube.com/watch?v=EhGBPqGKBl0',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  nojang: {
    name: 'nojang',
    koreanName: '노장',
    origin: '봉산탈춤',
    description: '늙은 고승을 상징하는 탈입니다. 도를 닦은 듯 하지만 결국 세속적 욕망에 빠지는 위선적 모습을 보여줍니다.',
    danceDescription: '노장은 소무의 아름다움에 반해 파계하는 장면을 연기하며, 종교적 권위의 허상을 풍자합니다.',
    character: '위선적인 늙은 승려',
    videoUrl: 'https://www.youtube.com/watch?v=EhGBPqGKBl0',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  somu: {
    name: 'somu',
    koreanName: '소무',
    origin: '봉산탈춤',
    description: '젊은 무희를 상징하는 탈로, 아름답고 요염한 표정이 특징입니다.',
    danceDescription: '소무는 노장을 유혹하여 파계하게 만드는 역할을 하며, 화려한 춤사위로 관객을 매료시킵니다.',
    character: '아름답고 요염한 무희',
    videoUrl: 'https://www.youtube.com/watch?v=EhGBPqGKBl0',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  yeongam: {
    name: 'yeongam',
    koreanName: '영감',
    origin: '봉산탈춤',
    description: '늙은 남편을 상징하는 탈로, 첩을 얻어 본처를 홀대하는 가부장적 인물입니다.',
    danceDescription: '영감은 미얄할미와 덜머리집 사이에서 갈등하는 장면을 통해 조선시대 가정 내 갈등을 보여줍니다.',
    character: '가부장적인 늙은 남편',
    videoUrl: 'https://www.youtube.com/watch?v=EhGBPqGKBl0',
    wikiUrl: 'https://ko.wikipedia.org/wiki/봉산탈춤'
  },
  
  // 고성오광대
  mundungi: {
    name: 'mundungi',
    koreanName: '문둥이',
    origin: '고성오광대',
    description: '문둥병 환자를 상징하는 탈로, 병으로 일그러진 얼굴이 특징입니다. 사회적 약자의 한을 표현합니다.',
    danceDescription: '문둥이 과장에서는 병자의 슬픔과 한을 춤으로 표현하며, 관객에게 연민과 카타르시스를 선사합니다.',
    character: '사회적 약자의 한',
    videoUrl: 'https://www.youtube.com/watch?v=TcvWJkKbVKo',
    wikiUrl: 'https://ko.wikipedia.org/wiki/고성오광대'
  },
  bibi: {
    name: 'bibi',
    koreanName: '비비',
    origin: '고성오광대',
    description: '양반의 첩을 상징하는 탈로, 요염하고 교태로운 표정이 특징입니다.',
    danceDescription: '비비는 양반과 말뚝이 사이의 갈등에서 중요한 역할을 하며, 화려한 춤으로 무대를 장식합니다.',
    character: '요염한 양반의 첩',
    videoUrl: 'https://www.youtube.com/watch?v=TcvWJkKbVKo',
    wikiUrl: 'https://ko.wikipedia.org/wiki/고성오광대'
  },
  joje: {
    name: 'joje',
    koreanName: '조제',
    origin: '고성오광대',
    description: '제비를 상징하는 탈로, 흥부놀부전의 제비를 연상시키는 독특한 형태입니다.',
    danceDescription: '조제 과장에서는 제비의 날렵한 움직임을 춤으로 표현합니다.',
    character: '날렵한 제비',
    videoUrl: 'https://www.youtube.com/watch?v=TcvWJkKbVKo',
    wikiUrl: 'https://ko.wikipedia.org/wiki/고성오광대'
  },
  
  // 특수 탈
  cheoyong: {
    name: 'cheoyong',
    koreanName: '처용',
    origin: '처용무',
    description: '신라 헌강왕 때의 인물 처용을 형상화한 탈입니다. 역신을 물리치는 벽사의 의미를 가지며, 궁중무용인 처용무에 사용됩니다. 붉은 얼굴에 미소 짓는 표정이 특징입니다.',
    danceDescription: '처용무는 유네스코 인류무형문화유산으로, 다섯 명의 무용수가 오방색 의상을 입고 추는 궁중무용입니다. 역신을 물리치고 나라의 태평을 기원하는 의미를 담고 있습니다.',
    character: '역신을 물리치는 벽사의 존재',
    videoUrl: 'https://www.youtube.com/watch?v=8IyVGZvBGrk',
    wikiUrl: 'https://ko.wikipedia.org/wiki/처용무'
  },
  bangsangsi: {
    name: 'bangsangsi',
    koreanName: '방상씨',
    origin: '나례',
    description: '귀신을 쫓는 벽사 의식에 사용되는 탈입니다. 네 개의 황금빛 눈이 특징이며, 곰가죽 옷을 입고 창과 방패를 들고 귀신을 쫓습니다.',
    danceDescription: '방상씨는 궁중의 나례 의식에서 역귀를 쫓는 역할을 했습니다. 위엄 있는 동작으로 사방을 누비며 악귀를 물리칩니다.',
    character: '귀신을 쫓는 벽사의 신',
    videoUrl: 'https://www.youtube.com/watch?v=5ydvSYqkj9s',
    wikiUrl: 'https://ko.wikipedia.org/wiki/방상시'
  },
  
  // 기타 유명 탈
  songpa: {
    name: 'songpa',
    koreanName: '송파산대놀이탈',
    origin: '송파산대놀이',
    description: '서울 송파 지역의 산대놀이에 사용되는 탈입니다. 서민적이고 해학적인 표정이 특징입니다.',
    danceDescription: '송파산대놀이는 서울 지역의 대표적인 탈놀이로, 양반 풍자와 파계승 이야기 등 다양한 과장으로 구성됩니다.',
    character: '서민적 해학',
    videoUrl: 'https://www.youtube.com/watch?v=QVgQvGXXlB0',
    wikiUrl: 'https://ko.wikipedia.org/wiki/송파산대놀이'
  },
  gangnyeong: {
    name: 'gangnyeong',
    koreanName: '강령탈',
    origin: '강령탈춤',
    description: '황해도 강령 지역의 탈춤에 사용되는 탈입니다. 봉산탈춤과 비슷하지만 더 투박하고 거친 느낌이 특징입니다.',
    danceDescription: '강령탈춤은 봉산탈춤과 함께 황해도를 대표하는 탈놀이로, 역동적인 춤사위가 특징입니다.',
    character: '투박하고 역동적인 서민',
    videoUrl: 'https://www.youtube.com/watch?v=UBmVBQkZ3T8',
    wikiUrl: 'https://ko.wikipedia.org/wiki/강령탈춤'
  },
  tongyeong: {
    name: 'tongyeong',
    koreanName: '통영오광대탈',
    origin: '통영오광대',
    description: '경남 통영 지역의 오광대놀이에 사용되는 탈입니다. 남해안 특유의 개방적이고 활달한 분위기가 느껴집니다.',
    danceDescription: '통영오광대는 경남 지역의 대표적인 탈놀이로, 바다와 인접한 지역 특성상 활기차고 역동적인 공연이 특징입니다.',
    character: '활달하고 개방적인 남해안 기질',
    videoUrl: 'https://www.youtube.com/watch?v=ZVmCE_S3JYU',
    wikiUrl: 'https://ko.wikipedia.org/wiki/통영오광대'
  },
  suyeong: {
    name: 'suyeong',
    koreanName: '수영야류탈',
    origin: '수영야류',
    description: '부산 수영 지역의 야류(들놀이)에 사용되는 탈입니다. 영남 지역 특유의 강인한 느낌이 특징입니다.',
    danceDescription: '수영야류는 부산 지역을 대표하는 민속놀이로, 정월대보름에 행해지던 마을 축제입니다.',
    character: '강인한 영남의 기질',
    videoUrl: 'https://www.youtube.com/watch?v=BnKpJLfLcVU',
    wikiUrl: 'https://ko.wikipedia.org/wiki/수영야류'
  }
};

// 탈 이름으로 정보 찾기 (부분 매칭 지원)
export function findMaskInfo(maskName: string): MaskDetailInfo | null {
  const normalizedName = maskName.toLowerCase().replace(/[^a-z가-힣]/g, '');
  
  // 정확한 매칭
  if (MASK_INFO[normalizedName]) {
    return MASK_INFO[normalizedName];
  }
  
  // 부분 매칭 (탈 이름이 포함된 경우)
  for (const [key, info] of Object.entries(MASK_INFO)) {
    if (normalizedName.includes(key) || key.includes(normalizedName) ||
        normalizedName.includes(info.koreanName) || info.koreanName.includes(normalizedName)) {
      return info;
    }
  }
  
  return null;
}
